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

    param = rsg::make_tensor_tree();
    tensor_tree::load_tensor(param, args.at("param"));

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

            la::vector<double> label_vec;
            label_vec.resize(id_label.size());

            label_vec(label_id.at(segs.at(index).label)) = 1;

            auto output = rsg::make_nn(comp_graph.var(la::tensor<double>(label_vec)),
                init_input, var_tree, end_time - start_time);

            // std::vector<std::shared_ptr<autodiff::op_t>> seg_frames;

            // for (int i = start_time; i < end_time; ++i) {
            //     seg_frames.push_back(comp_graph.var(la::tensor<double>(la::vector<double>(frames.at(i)))));
            // }

            // auto output = rsg::make_training_nn(comp_graph.var(la::tensor<double>(label_vec)),
            //      seg_frames, var_tree);

            auto topo_order = autodiff::topo_order(output);

            autodiff::eval(topo_order, autodiff::eval_funcs);

            for (int t = 1; t < output.size(); ++t) {
                auto& v = autodiff::get_output<la::tensor<double>>(output.at(t-1));

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

