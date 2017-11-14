#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include <vector>
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include <algorithm>

std::vector<std::shared_ptr<autodiff::op_t>> make_autoenc(std::shared_ptr<autodiff::op_t> input,
    std::shared_ptr<tensor_tree::vertex> var_tree)
{
    std::vector<std::shared_ptr<autodiff::op_t>> result;

    std::shared_ptr<autodiff::op_t> h = input;

    for (int i = 0; i < var_tree->children[0]->children.size(); ++i) {
        auto z = autodiff::mul(h, tensor_tree::get_var(
            var_tree->children[0]->children[i]->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(
            var_tree->children[0]->children[i]->children[1]), z);
        h = autodiff::relu(autodiff::add(z, b));

        result.push_back(h);
    }

    for (int i = 0; i < var_tree->children[1]->children.size(); ++i) {
        auto z = autodiff::mul(h, tensor_tree::get_var(
            var_tree->children[1]->children[i]->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(
            var_tree->children[1]->children[i]->children[1]), z);

        h = autodiff::add(z, b);

        if (i != var_tree->children[1]->children.size() - 1) {
            h = autodiff::relu(h);
        }

        result.push_back(h);
    }

    return result;
}

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    tensor_tree::vertex root { "nil" };

    tensor_tree::vertex encoder { "nil" };
    for (int i = 0; i < layer; ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("encoder weight"));
        ell.children.push_back(tensor_tree::make_tensor("encoder bias"));
        encoder.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }
    root.children.push_back(std::make_shared<tensor_tree::vertex>(encoder));

    tensor_tree::vertex decoder { "nil" };
    for (int i = 0; i < layer; ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("decoder weight"));
        ell.children.push_back(tensor_tree::make_tensor("decoder bias"));
        decoder.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }
    root.children.push_back(std::make_shared<tensor_tree::vertex>(decoder));

    return std::make_shared<tensor_tree::vertex>(root);
}

struct prediction_env {

    batch::scp input_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "fc-recon",
        "Train a 1-layer NN autoencoder",
        {
            {"input-scp", "", true},
            {"param", "", true},
            {"eval", "", false}
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
    input_scp.open(args.at("input-scp"));

    std::ifstream param_ifs { args.at("param") };
    if (!param_ifs) {
        throw std::logic_error("failed to open " + args.at("param"));
    }
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);

    param = make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();
}

void prediction_env::run()
{
    int nsample = 0;

    while (nsample < input_scp.entries.size()) {
        std::vector<double> raw_input = batch::load_feats(input_scp.at(nsample));

        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        la::cpu::tensor<double> input_tensor { la::cpu::vector<double> { raw_input } };

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        auto layers = make_autoenc(input, var_tree);

        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(layers.back());

        if (ebt::in(std::string("eval"), args)) {
            la::cpu::weak_tensor<double> gold { input_tensor.data(), input_tensor.sizes() };

            nn::l2_loss loss { gold, pred };

            double ell = loss.loss();

            std::cout << "loss: " << ell << std::endl;
        } else {
            std::cout << input_scp.entries[nsample].key << std::endl;

            for (int i = 0; i < pred.vec_size(); ++i) {
                if (i != 0) {
                    std::cout << " ";
                }

                std::cout << pred.data()[i];
            }

            std::cout << std::endl;
            std::cout << "." << std::endl;
        }

        ++nsample;
    }
}

