#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/cnn.h"
#include "nn/cnn-frame.h"
#include <algorithm>

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    tensor_tree::vertex root { "nil" };

    for (int i = 0; i < layer; ++i) {
        tensor_tree::vertex fc;
        fc.children.push_back(tensor_tree::make_tensor("fc weight"));
        fc.children.push_back(tensor_tree::make_tensor("fc bias"));
        root.children.push_back(std::make_shared<tensor_tree::vertex>(fc));
    }

    tensor_tree::vertex pred;
    pred.children.push_back(tensor_tree::make_tensor("fc weight"));
    pred.children.push_back(tensor_tree::make_tensor("fc bias"));
    root.children.push_back(std::make_shared<tensor_tree::vertex>(pred));

    return std::make_shared<tensor_tree::vertex>(root);
}

std::shared_ptr<autodiff::op_t> make_nn(std::shared_ptr<autodiff::op_t> input,
    std::shared_ptr<tensor_tree::vertex> var_tree)
{
    std::shared_ptr<autodiff::op_t> h = input;

    for (int i = 0; i < var_tree->children.size() - 1; ++i) {
        auto z = autodiff::mul(h, tensor_tree::get_var(var_tree->children[i]->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[i]->children[1]), z);
        h = autodiff::logistic(autodiff::add(z, b));
    }

    auto z = autodiff::mul(h, tensor_tree::get_var(var_tree->children.back()->children[0]));
    auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children.back()->children[1]), z);

    return autodiff::add(z, b);
}

struct prediction_env {

    std::ifstream frame_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    unsigned int win_size;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-win-fc-autoenc-recon",
        "Reconstruct with an FC frame autoencoder",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"win-size", "", true},
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
    param = make_tensor_tree(1);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    win_size = std::stoi(args.at("win-size"));
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (true) {
        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        unsigned input_dim = frames.front().size();

        std::cout << nsample << ".logmel" << std::endl;

        for (int t = 0; t < frames.size(); ++t) {
            std::vector<double> input_tensor_vec;

            for (int i = 0; i < win_size; ++i) {
                if (0 <= t + i - win_size / 2 && t + i - win_size / 2 < frames.size()) {
                    for (int j = 0; j < input_dim; ++j) {
                        input_tensor_vec.push_back(frames[t + i - win_size / 2][j]);
                    }
                } else {
                    for (int j = 0; j < input_dim; ++j) {
                        input_tensor_vec.push_back(0);
                    }
                }
            }

            la::cpu::tensor<double> input_tensor { la::cpu::vector<double>(input_tensor_vec),
                std::vector<unsigned int> { 1, win_size * input_dim }};

            std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

            auto pred = make_nn(input, var_tree);

            auto& pred_t = autodiff::get_output<la::cpu::tensor_like<double>>(pred);

            int mid = win_size / 2 * input_dim;

            for (int i = 0; i < input_dim ; ++i) {
                if (i != 0) {
                    std::cout << " ";
                }

                std::cout << pred_t({0, mid + i});
            }

            std::cout << std::endl;
        }

        std::cout << "." << std::endl;

        ++nsample;
    }
}

