#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/tensor-tree.h"
#include "nn/rsg.h"
#include "nn/nn.h"
#include <random>
#include <algorithm>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream seg_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> const& args);

    void run();

};

learning_env::learning_env(std::unordered_map<std::string, std::string> const& args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));
    seg_batch.open(args.at("seg-batch"));

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);

    param = rsg::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "rsg-learn",
        "Train a Recurrent Sequence Generator",
        {
            {"frame-batch", "", true},
            {"seg-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"use-gt", "", false},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}

void learning_env::run()
{
    int nsample = 0;

    int err = 0;
    int total = 0;

    std::vector<double> err_by_label;
    err_by_label.resize(label_id.size());

    std::vector<double> total_by_label;
    total_by_label.resize(label_id.size());

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);
        std::vector<speech::segment> segs = speech::load_segment_batch(seg_batch);

        if (!frame_batch || !seg_batch) {
            break;
        }

        for (int index = 0; index < segs.size(); ++index) {

            int start_time = std::min<int>(frames.size() - 1,
                std::max<int>(0, segs.at(index).start_time));
            int end_time = std::min<int>(frames.size(),
                std::max<int>(0, segs.at(index).end_time));

            std::vector<double> id_loss;
            id_loss.resize(id_label.size());

            for (int id = 0; id < id_label.size(); ++id) {
                autodiff::computation_graph comp_graph;

                auto var_tree = tensor_tree::make_var_tree(comp_graph, param);

                std::vector<std::shared_ptr<autodiff::op_t>> seg_frames;

                for (int i = start_time; i < end_time - 1; ++i) {
                    seg_frames.push_back(comp_graph.var(la::tensor<double>(la::vector<double>(frames.at(i)))));
                }

                la::vector<double> label_vec;
                label_vec.resize(id_label.size());

                label_vec(id) = 1;

                if (seg_frames.size() <= 1) {
                    std::cout << std::endl;

                    continue;
                }

                lstm::lstm_multistep_transcriber multistep;
                multistep.steps.push_back(std::make_shared<lstm::dyer_lstm_step_transcriber>(
                    lstm::dyer_lstm_step_transcriber{}));

                std::shared_ptr<lstm::lstm_step_transcriber> step
                    = std::make_shared<lstm::lstm_multistep_transcriber>(multistep);

                std::vector<std::shared_ptr<autodiff::op_t>> output;

                if (ebt::in(std::string("use-gt"), args)) {
                    output = rsg::make_training_nn(comp_graph.var(la::tensor<double>(label_vec)),
                        seg_frames, var_tree, step);
                } else {
                    output = rsg::make_nn(comp_graph.var(la::tensor<double>(label_vec)),
                        seg_frames[0], var_tree, end_time - start_time - 1, step);
                }

                auto topo_order = autodiff::topo_order(output);

                autodiff::eval(topo_order, autodiff::eval_funcs);

                double loss_sum = 0;

                for (int t = 0; t < output.size(); ++t) {
                    la::tensor<double> gold { la::vector<double>(frames.at(t + start_time + 1)) };

                    nn::l2_loss loss {
                        gold,
                        autodiff::get_output<la::tensor_like<double>>(output.at(t))
                    };

                    loss_sum += loss.loss();

                    output.at(t)->grad = std::make_shared<la::tensor<double>>(loss.grad());
                }

                id_loss[id] = loss_sum;
            }

            double inf = std::numeric_limits<double>::infinity();
            double min = inf;
            int argmin = -1;

            for (int id = 0; id < id_loss.size(); ++id) {
                if (id_loss[id] < min) {
                    min = id_loss[id];
                    argmin = id;
                }
            }

            if (argmin != label_id.at(segs[index].label)) {
                err += 1;
                err_by_label[label_id.at(segs[index].label)] += 1;
            }

            total += 1;
            total_by_label[label_id.at(segs[index].label)] += 1;

            std::cout << "label: " << segs[index].label << std::endl;
            std::cout << "min loss label: " << id_label[argmin] << std::endl;
            std::cout << "loss: " << id_loss << std::endl;
            std::cout << std::endl;

        }

        ++nsample;
    }

    std::cout << "err: " << err_by_label << std::endl;
    std::cout << "total: " << total_by_label << std::endl;

    for (int i = 0; i < id_label.size(); ++i) {
        std::cout << "rate: " << double(err_by_label[i]) / total_by_label[i] << std::endl;
    }
    std::cout << "err: " << err << " total: " << total << " rate: " << double(err) / total << std::endl;

}

