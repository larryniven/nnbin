#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "util/speech.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include <vector>
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/cnn.h"
#include "nn/cnn-frame.h"
#include <algorithm>

struct prediction_env {

    std::ifstream frame_batch;

    unsigned int input_channel;

    std::shared_ptr<tensor_tree::vertex> param;

    cnn::cnn_t cnn_config;

    unsigned int win_size;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-win-cnn-predict",
        "Train a CNN frame classifier",
        {
            {"frame-batch", "", true},
            {"input-channel", "", true},
            {"param", "", true},
            {"win-size", "", true},
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

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));

    input_channel = std::stoi(args.at("input-channel"));

    std::ifstream param_ifs { args.at("param") };
    cnn_config = cnn::load_param(param_ifs);
    param = cnn_config.param;
    param_ifs.close();

    win_size = std::stoi(args.at("win-size"));

    id_label = util::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (true) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        unsigned int input_dim = frames.front().size() / input_channel;

        std::cout << nsample << ".blabel" << std::endl;

        for (int t = 0; t < frames.size(); ++t) {

            std::vector<double> input_tensor_vec;

            for (int i = 0; i < win_size; ++i) {
                if (0 <= t + i - win_size / 2 && t + i - win_size / 2 < frames.size()) {
                    for (int j = 0; j < frames.front().size(); ++j) {
                        input_tensor_vec.push_back(frames[t + i - win_size / 2][j]);
                    }
                } else {
                    for (int j = 0; j < frames.front().size(); ++j) {
                        input_tensor_vec.push_back(0);
                    }
                }
            }

            la::cpu::tensor<double> input_tensor { la::cpu::vector<double>(input_tensor_vec),
                std::vector<unsigned int> { 1, win_size, input_dim, input_channel }};

            std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

            std::shared_ptr<cnn::transcriber> trans = cnn::make_transcriber(cnn_config, 0.0, nullptr);
            std::shared_ptr<autodiff::op_t> logprob = (*trans)(var_tree, input);

            auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);

            double max = -std::numeric_limits<double>::infinity();
            int argmax = -1;

            for (int i = 0; i < id_label.size(); ++i) {
                if (pred({0, i}) > max) {
                    max = pred({0, i});
                    argmax = i;
                }
            }

            std::cout << id_label.at(argmax) << std::endl;

        }

        std::cout << std::endl;

        ++nsample;
    }
}

