#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/lstm.h"
#include "nn/tensor-tree.h"
#include "nn/lstm-frame.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> label;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-lstm-predict",
        "Predict frames with LSTM",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"logprob", "", false},
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
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = lstm_frame::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    label = speech::load_label_set(args.at("label"));
}

void prediction_env::run()
{
    int nsample = 1;

    while (1) {
        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph graph;
        std::vector<std::shared_ptr<autodiff::op_t>> inputs;

        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(graph.var(la::tensor<double>(la::vector<double>(frames[i]))));
        }

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::shared_ptr<lstm::transcriber> trans
            = lstm_frame::make_transcriber(layer, 0.0, nullptr);

        std::vector<std::shared_ptr<autodiff::op_t>> output;

        if (ebt::in(std::string("print-hidden"), args)) {
            output = (*trans)(var_tree->children[0], inputs);
        } else {
            trans = std::make_shared<lstm::logsoftmax_transcriber>(
                lstm::logsoftmax_transcriber { trans });
            output = (*trans)(var_tree, inputs);
        }

        std::cout << nsample << ".phn" << std::endl;

        if (ebt::in(std::string("print-logprob"), args) || ebt::in(std::string("print-hidden"), args)) {
            for (int t = 0; t < output.size(); ++t) {
                auto& pred = autodiff::get_output<la::tensor_like<double>>(output[t]);

                std::cout << pred({0});

                for (int j = 1; j < pred.vec_size(); ++j) {
                    std::cout << " " << pred({j});
                }

                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < output.size(); ++t) {
                auto& pred = autodiff::get_output<la::tensor_like<double>>(output[t]);

                int argmax = -1;
                double max = -std::numeric_limits<double>::infinity();

                for (int j = 0; j < pred.vec_size(); ++j) {
                    if (pred({j}) > max) {
                        max = pred({j});
                        argmax = j;
                    }
                }

                std::cout << label[argmax] << std::endl;
            }
        }

        std::cout << "." << std::endl;

        ++nsample;
    }
}

