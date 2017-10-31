#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/nn.h"
#include "nn/tensor-tree.h"
#include "nn/lstm.h"
#include "nn/lstm-tensor-tree.h"
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    lstm::multilayer_lstm_tensor_tree_factory lstm_factory {
        std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
            lstm::bi_lstm_tensor_tree_factory {
                std::make_shared<lstm::lstm_tensor_tree_factory>(
                    lstm::lstm_tensor_tree_factory{})
            }),
        layer
    };

    tensor_tree::vertex root { "nil" };

    root.children.push_back(lstm_factory());
    root.children.push_back(lstm_factory());
    root.children.push_back(tensor_tree::make_tensor("regression weight"));
    root.children.push_back(tensor_tree::make_tensor("regression bias"));

    return std::make_shared<tensor_tree::vertex>(root);
}

std::shared_ptr<lstm::transcriber>
make_transcriber(
    std::shared_ptr<tensor_tree::vertex> param,
    double dropout,
    std::default_random_engine *gen,
    bool pyramid)
{
    int layer = param->children.size();

    lstm::layered_transcriber trans;

    for (int i = 0; i < layer; ++i) {
        std::shared_ptr<lstm::transcriber> f_trans;
        std::shared_ptr<lstm::transcriber> b_trans;

        f_trans = std::make_shared<lstm::lstm_transcriber>(
            lstm::lstm_transcriber { (int) tensor_tree::get_tensor(
                param->children[i]->children[0]->children[2]).size(0) });
        b_trans = std::make_shared<lstm::lstm_transcriber>(
            lstm::lstm_transcriber { (int) tensor_tree::get_tensor(
                param->children[i]->children[1]->children[2]).size(0), true });

        if (dropout != 0.0) {
            f_trans = std::make_shared<lstm::input_dropout_transcriber>(
                lstm::input_dropout_transcriber { f_trans, dropout, *gen });

            b_trans = std::make_shared<lstm::input_dropout_transcriber>(
                lstm::input_dropout_transcriber { b_trans, dropout, *gen });
        }

        trans.layer.push_back(std::make_shared<lstm::bi_transcriber>(
            lstm::bi_transcriber { (int) tensor_tree::get_tensor(
                param->children[i]->children[2]).size(1), f_trans, b_trans }));
    }

    return std::make_shared<lstm::layered_transcriber>(trans);
}

struct prediction_env {

    speech::scp frame_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    double input_dropout;

    int seed;
    std::default_random_engine gen;

    std::vector<int> indices;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "utt-autoenc-lstm-recon",
        "Reconstruct with an lstm autoencoder",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"input-dropout", "", false},
            {"seed", "", false},
            {"print-hidden", "", false},
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

    std::string line;
    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    input_dropout = 0;
    if (ebt::in(std::string("input-dropout"), args)) {
        input_dropout = std::stod(args.at("input-dropout"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    gen = std::default_random_engine { seed };

    for (int i = 0; i < frame_scp.entries.size(); ++i) {
        indices.push_back(i);
    }
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < indices.size()) {
        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::vector<std::vector<double>> frames = speech::load_frame_batch(
            frame_scp.at(indices.at(nsample)));

        int input_dim = frames.front().size();

        std::vector<double> input_vec;

        for (int i = 0; i < frames.size(); ++i) {
            for (int j = 0; j < input_dim; ++j) {
                input_vec.push_back(frames[i][j]);
            }
        }

        la::cpu::tensor<double> input_t { la::cpu::vector<double>(input_vec),
            { (unsigned int) frames.size(), 1, (unsigned int) input_dim }};

        auto input_var = graph.var(input_t);

        lstm::trans_seq_t seq = lstm::make_trans_seq(input_var);

        std::shared_ptr<lstm::transcriber> enc_trans
            = make_transcriber(param->children[0], input_dropout, &gen, false);

        lstm::trans_seq_t h1 = (*enc_trans)(var_tree->children[0], seq);

        std::shared_ptr<lstm::transcriber> dec_trans
            = make_transcriber(param->children[1], 0.0, nullptr, false);
        
        lstm::trans_seq_t h2 = (*dec_trans)(var_tree->children[1], h1);

        auto z = autodiff::mul(h2.feat, tensor_tree::get_var(var_tree->children[2]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[3]), z);
        auto pred = autodiff::add(z, b);

        if (ebt::in(std::string("print-hidden"), args)) {
            auto& h1_t = autodiff::get_output<la::cpu::tensor_like<double>>(h1.feat);

            std::cerr << h1_t.sizes() << std::endl;

            std::cout << frame_scp.entries[nsample].key << std::endl;

            for (int i = 0; i < h1_t.size(0); ++i) {
                for (int j = 0; j < h1_t.size(2); ++j) {
                    std::cout << h1_t({i, 0, j}) << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "." << std::endl;
        } else {
            auto& pred_t = autodiff::get_output<la::cpu::tensor_like<double>>(pred);

            std::cerr << pred_t.sizes() << std::endl;

            std::cout << frame_scp.entries[nsample].key << std::endl;

            for (int i = 0; i < pred_t.size(0); ++i) {
                for (int j = 0; j < pred_t.size(2); ++j) {
                    std::cout << pred_t({i, 0, j}) << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "." << std::endl;
        }

        ++nsample;

    }
}

