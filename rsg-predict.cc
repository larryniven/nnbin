#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/tensor-tree.h"
#include "nn/rsg.h"
#include "nn/nn.h"
#include <random>
#include <algorithm>

struct prediction_env {

    std::ifstream frame_batch;
    std::ifstream seg_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> const& args);

    void run();

};

prediction_env::prediction_env(std::unordered_map<std::string, std::string> const& args)
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
    param_ifs.close();

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

    if (ebt::in(std::string("clamp"), args)) {
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
            {"clamp", "", false}
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

    prediction_env env { args };

    env.run();

    return 0;
}

void prediction_env::run()
{
    int nsample = 0;

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);
        std::vector<speech::segment> segs = speech::load_segment_batch(seg_batch);

        if (!frame_batch || !seg_batch) {
            break;
        }

        std::cout << nsample << ".logmel" << std::endl;

        for (int index = 0; index < segs.size(); ++index) {
            autodiff::computation_graph comp_graph;

            auto var_tree = tensor_tree::make_var_tree(comp_graph, param);

            int start_time = std::min<int>(frames.size() - 1,
                std::max<int>(0, segs.at(index).start_time));
            int end_time = std::min<int>(frames.size(),
                std::max<int>(0, segs.at(index).end_time));

            auto& init_frame = frames.at(start_time);

            for (int d = 0; d < init_frame.size(); ++d) {
                std::cout << init_frame[d];

                if (d != init_frame.size() - 1) {
                    std::cout << " ";
                }
            }

            std::cout << std::endl;

            if (end_time - start_time == 1) {
                continue;
            }

            auto init_input = comp_graph.var(la::tensor<double>(
                la::vector<double>(frames.at(start_time))));

            lstm::lstm_multistep_transcriber multistep;
            multistep.steps.push_back(std::make_shared<lstm::dyer_lstm_step_transcriber>(
                lstm::dyer_lstm_step_transcriber{}));

            la::vector<double> label_vec;
            label_vec.resize(id_label.size());
            label_vec(label_id.at(segs.at(index).label)) = 1;

            auto label_embed = autodiff::mul(comp_graph.var(la::tensor<double>(label_vec)),
                tensor_tree::get_var(var_tree->children[0]));

            std::shared_ptr<autodiff::op_t> frame = init_input;

            std::vector<std::shared_ptr<autodiff::op_t>> outputs;

            int frames = end_time - start_time - 1;

            for (int i = 0; i < frames; ++i) {
                la::vector<double> dur_vec;
                dur_vec.resize(100);
                dur_vec(frames - i) = 1;

                auto dur_embed = autodiff::mul(comp_graph.var(la::tensor<double>(dur_vec)),
                    tensor_tree::get_var(var_tree->children[1]));
                auto acoustic_embed = autodiff::mul(frame,
                    tensor_tree::get_var(var_tree->children[2]));

                auto input_embed = autodiff::add(
                    std::vector<std::shared_ptr<autodiff::op_t>>{ label_embed,
                        dur_embed, acoustic_embed,
                        tensor_tree::get_var(var_tree->children[3]) });

                auto output = multistep(var_tree->children[4], input_embed);

                outputs.push_back(autodiff::add(autodiff::mul(output,
                    tensor_tree::get_var(var_tree->children[5])),
                    tensor_tree::get_var(var_tree->children[6])));

                frame = outputs.back();
            }

            for (int t = 0; t < outputs.size(); ++t) {
                auto& v = autodiff::get_output<la::tensor<double>>(outputs.at(t));

                for (int d = 0; d < v.vec_size(); ++d) {
                    std::cout << v.data()[d];

                    if (d != v.vec_size() - 1) {
                        std::cout << " ";
                    }
                }

                std::cout << std::endl;
            }
        }

        std::cout << "." << std::endl;

        ++nsample;
    }

}

