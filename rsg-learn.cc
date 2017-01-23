#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/tensor-tree.h"
#include "nn/rsg.h"
#include "nn/nn.h"
#include <random>
#include <algorithm>

struct learning_env {

    speech::batch_indices frame_batch;
    speech::batch_indices seg_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::string output_param;
    std::string output_opt_data;

    std::vector<int> sample_indices;

    double step_size;
    double decay;
    double clip;

    double dropout;

    std::default_random_engine gen;
    int seed;

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

    param = rsg::make_tensor_tree();
    tensor_tree::load_tensor(param, args.at("param"));

    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    if (ebt::in(std::string("step-size"), args)) {
        step_size = std::stod(args.at("step-size"));
    }

    decay = 0;
    if (ebt::in(std::string("decay"), args)) {
        decay = std::stod(args.at("decay"));
    }

    clip = 0;
    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    seed = 0;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

    if (ebt::in(std::string("opt-data"), args)) {
        if (ebt::in(std::string("decay"), args)) {
            opt = std::make_shared<tensor_tree::rmsprop_opt>(
                tensor_tree::rmsprop_opt(param, decay, step_size));
        } else if (ebt::in(std::string("const-step-update"), args)) {
            opt = std::make_shared<tensor_tree::const_step_opt>(
                tensor_tree::const_step_opt(param, step_size));
        } else {
            opt = std::make_shared<tensor_tree::adagrad_opt>(
                tensor_tree::adagrad_opt(param, step_size));
        }
    }

    if (ebt::in(std::string("opt-data"), args)) {
        std::ifstream opt_data_ifs { args.at("opt-data") };
        opt->load_opt_data(opt_data_ifs);
        opt_data_ifs.close();
    }

    gen = std::default_random_engine { seed };

    sample_indices.resize(frame_batch.pos.size());

    for (int i = 0; i < sample_indices.size(); ++i) {
        sample_indices[i] = i;
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::shuffle(sample_indices.begin(), sample_indices.end(), gen);

        std::vector<unsigned long> pos = frame_batch.pos;
        for (int i = 0; i < sample_indices.size(); ++i) {
            frame_batch.pos[i] = pos[sample_indices[i]];
        }

        pos = seg_batch.pos;
        for (int i = 0; i < sample_indices.size(); ++i) {
            seg_batch.pos[i] = pos[sample_indices[i]];
        }
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
            {"opt-data", "", false},
            {"step-size", "", false},
            {"decay", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"clip", "", false},
            {"label", "", true},
            {"dropout", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"const-step-update", "", false},
            {"use-gt", "", false},
            {"loss-only", "", false},
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

    while (nsample < frame_batch.pos.size()) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));
        std::vector<speech::segment> segs = speech::load_segment_batch(seg_batch.at(nsample));

        std::vector<int> indices;
        for (int i = 0; i < segs.size(); ++i) {
            indices.push_back(i);
        }

        if (ebt::in(std::string("shuffle"), args)) {
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        for (int i = 0; i < segs.size(); ++i) {
            int index = indices.at(i);

            std::cout << "sample: " << sample_indices[nsample] << " seg: " << index << std::endl;

            autodiff::computation_graph comp_graph;

            auto var_tree = tensor_tree::make_var_tree(comp_graph, param);

            int start_time = std::min<int>(frames.size() - 1,
                std::max<int>(0, segs.at(index).start_time));
            int end_time = std::min<int>(frames.size(),
                std::max<int>(0, segs.at(index).end_time));

            std::vector<std::shared_ptr<autodiff::op_t>> seg_frames;

            for (int i = start_time; i < end_time - 1; ++i) {
                seg_frames.push_back(comp_graph.var(la::tensor<double>(la::vector<double>(frames.at(i)))));
            }

            la::vector<double> label_vec;
            label_vec.resize(id_label.size());

            label_vec(label_id.at(segs.at(index).label)) = 1;

            if (seg_frames.size() <= 1) {
                std::cout << std::endl;

                continue;
            }

            std::vector<std::shared_ptr<autodiff::op_t>> output;

            if (ebt::in(std::string("use-gt"), args)) {
                output = rsg::make_training_nn(comp_graph.var(la::tensor<double>(label_vec)),
                    seg_frames, var_tree);
            } else {
                output = rsg::make_nn(comp_graph.var(la::tensor<double>(label_vec)),
                    seg_frames[0], var_tree, end_time - start_time - 1);
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

            std::cout << "label: " << segs.at(index).label << std::endl;
            std::cout << "frames: " << seg_frames.size() << std::endl;
            std::cout << "loss: " << loss_sum / seg_frames.size() << std::endl;

            if (ebt::in(std::string("loss-only"), args)) {
                std::cout << std::endl;

                continue;
            }

            autodiff::grad(topo_order, autodiff::grad_funcs);

            auto grad = rsg::make_tensor_tree();

            tensor_tree::copy_grad(grad, var_tree);

            double n = tensor_tree::norm(grad);

            std::cout << "grad norm: " << n << std::endl;

            if (ebt::in(std::string("clip"), args)) {
                if (n > clip) {
                    tensor_tree::imul(grad, clip / n);
                    std::cout << "gradient clipped" << std::endl;
                }
            }

            auto vars = tensor_tree::leaves_pre_order(param);
            la::tensor<double> const& v = tensor_tree::get_tensor(vars[2]);

            double v1 = v.data()[0];

            opt->update(grad);

            double v2 = v.data()[0];

            std::cout << "weight: " << v1 << " update: " << v2 - v1
                << " rate: " << (v2 - v1) / v1 << std::endl;

            std::cout << std::endl;
        }

        ++nsample;
    }

    if (!ebt::in(std::string("loss-only"), args)) {
        tensor_tree::save_tensor(param, output_param);

        std::ofstream opt_data_ofs { output_opt_data };
        opt->save_opt_data(opt_data_ofs);
        opt_data_ofs.close();
    }

}

