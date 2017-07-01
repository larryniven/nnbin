#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/lstm.h"
#include "nn/tensor-tree.h"
#include "nn/lstm-frame.h"
#include <fstream>

std::shared_ptr<tensor_tree::vertex> make_tensor_tree()
{
    tensor_tree::vertex root { "nil" };

    root.children.push_back(tensor_tree::make_tensor("dummy"));
    root.children.push_back(tensor_tree::make_tensor("weight"));
    root.children.push_back(tensor_tree::make_tensor("bias"));

    return std::make_shared<tensor_tree::vertex>(root);
}

struct prediction_env {

    std::ifstream frame_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> label;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-lin-predict",
        "Predict frames with LSTM",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"print-logprob", "", false},
            {"nsample", "", false},
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

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));

    std::ifstream param_ifs { args.at("param") };
    param = make_tensor_tree();
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    label = speech::load_label_set(args.at("label"));
}

void prediction_env::run()
{
    int nsample = 1;

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph graph;
        std::vector<double> input_vec;
        input_vec.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            input_vec.insert(input_vec.end(), frames[i].begin(), frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::cpu::weak_tensor<double>(input_vec.data(),
            {(unsigned int) frames.size(), (unsigned int) frames.front().size()}));

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::shared_ptr<autodiff::op_t> output;
        std::shared_ptr<autodiff::op_t> ignore;

        std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::logsoftmax_transcriber>(
            lstm::logsoftmax_transcriber { nullptr });
        std::tie(output, ignore) = (*trans)(var_tree, input);

        std::cout << nsample << ".phn" << std::endl;

        auto& output_t = autodiff::get_output<la::cpu::tensor_like<double>>(output);

        if (ebt::in(std::string("print-logprob"), args)) {
            for (int t = 0; t < output_t.size(0); ++t) {
                std::cout << output_t({t, 0});

                for (int j = 1; j < output_t.size(1); ++j) {
                    std::cout << " " << output_t({t, j});
                }

                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < output_t.size(0); ++t) {
                int argmax = -1;
                double max = -std::numeric_limits<double>::infinity();

                for (int j = 0; j < output_t.size(1); ++j) {
                    if (output_t({t, j}) > max) {
                        max = output_t({t, j});
                        argmax = j;
                    }
                }

                std::cout << label[argmax] << std::endl;
            }
        }

        std::cout << "." << std::endl;

        if (ebt::in(std::string("nsample"), args) && nsample >= std::stoi(args.at("nsample"))) {
            break;
        }

        ++nsample;
    }
}

