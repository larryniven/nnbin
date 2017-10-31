#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/nn.h"
#include "nn/tensor-tree.h"
#include "nn/autoenc-fc.h"
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>

struct prediction_env {

    speech::scp frame_scp;

    std::shared_ptr<tensor_tree::vertex> param;

    int print_channel;

    int patch_time;
    int patch_freq;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "utt-autoenc-patch-recon",
        "Reconstruct input with a patch autoencoder",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"patch-time", "", false},
            {"patch-freq", "", false},
            {"print-channel", "", false}
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
    frame_scp.open(args.at("frame-scp"));

    std::ifstream param_ifs { args.at("param") };
    param = autoenc::make_symmetric_ae_tensor_tree();
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    patch_time = 5;
    if (ebt::in(std::string("patch-time"), args)) {
        patch_time = std::stoi(args.at("patch-time"));
    }

    patch_freq = 5;
    if (ebt::in(std::string("patch-freq"), args)) {
        patch_freq = std::stoi(args.at("patch-freq"));
    }
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_scp.entries.size()) {
        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::vector<std::vector<double>> frames = speech::load_frame_batch(
            frame_scp.at(nsample));

        int input_dim = frames.front().size();

        std::vector<double> input_vec;

        for (int i = 0; i < frames.size(); ++i) {
            for (int j = 0; j < input_dim; ++j) {
                input_vec.push_back(frames[i][j]);
            }
        }

        la::cpu::tensor<double> input_tensor { la::cpu::vector<double>(input_vec),
            { 1, (unsigned int) frames.size(), (unsigned int) input_dim, 1 }};

        la::cpu::tensor<double> input_corr_lin;
        input_corr_lin.resize({(unsigned int) frames.size(), (unsigned int) input_dim,
            (unsigned int) patch_time * patch_freq});

        la::cpu::corr_linearize(input_corr_lin, input_tensor, patch_time, patch_freq,
            patch_time / 2, patch_freq / 2, 1, 1);

        auto input = graph.var(input_corr_lin);

        auto res = autodiff::mul(input, get_var(var_tree->children[0]));

        auto& res_t = autodiff::get_output<la::cpu::tensor<double>>(res);

        if (ebt::in(std::string("print-channel"), args)) {

            std::cerr << res_t.sizes() << std::endl;

            print_channel = std::stoi(args.at("print-channel"));

            std::cout << frame_scp.entries[nsample].key << std::endl;
            for (int i = 0; i < res_t.size(0); ++i) {
                for (int j = 0; j < res_t.size(1); ++j) {
                    std::cout << res_t({i, j, print_channel}) << " ";
                }

                std::cout << std::endl;
            }

            std::cout << "." << std::endl;

        } else {

            std::shared_ptr<autodiff::op_t> pred = autoenc::make_symmetric_ae(
                input, var_tree, 0.0, 0.0, nullptr);

            auto& pred_t = autodiff::get_output<la::cpu::tensor_like<double>>(pred);

            la::cpu::tensor<double> input_recon;
            input_recon.resize({ 1, (unsigned int) frames.size(), (unsigned int) input_dim, 1 });

            la::cpu::corr_delinearize(input_recon, pred_t, patch_time, patch_freq,
                patch_time / 2, patch_freq / 2, 1, 1);

            std::cout << frame_scp.entries[nsample].key << std::endl;
            for (int i = 0; i < frames.size(); ++i) {
                for (int j = 0; j < input_dim; ++j) {
                    std::cout << input_recon({0, i, j, 0}) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "." << std::endl;

        }

        ++nsample;

    }
}

