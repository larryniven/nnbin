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

std::vector<std::shared_ptr<autodiff::op_t>> make_fc(std::shared_ptr<autodiff::op_t> input,
    std::shared_ptr<tensor_tree::vertex> var_tree)
{
    std::vector<std::shared_ptr<autodiff::op_t>> result;

    std::shared_ptr<autodiff::op_t> h = input;

    for (int i = 0; i < var_tree->children.size(); ++i) {
        if (i != 0) {
            h = autodiff::relu(h);
        }

        auto z = autodiff::mul(h, tensor_tree::get_var(var_tree->children[i]->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(var_tree->children[i]->children[1]), z);

        h = autodiff::add(z, b);

        result.push_back(h);
    }

    return result;
}

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    tensor_tree::vertex root { "nil" };

    for (int i = 0; i < layer; ++i) {
        tensor_tree::vertex ell { "nil" };

        ell.children.push_back(tensor_tree::make_tensor("fc weight"));
        ell.children.push_back(tensor_tree::make_tensor("fc bias"));

        root.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }

    return std::make_shared<tensor_tree::vertex>(root);
}

struct prediction_env {

    batch::scp input_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<int> indices;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "fc-predict",
        "Predict with a DNN frame classifier",
        {
            {"input-scp", "", true},
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

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    input_scp.open(args.at("input-scp"));

    std::string line;
    std::ifstream param_ifs { args.at("param") };
    if (!param_ifs) {
        throw std::logic_error("failed to open " + args.at("param"));
    }

    std::getline(param_ifs, line);
    layer = std::stoi(line);

    param = make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    id_label = util::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

    indices.resize(input_scp.entries.size());

    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < indices.size()) {
        std::vector<double> raw_input = batch::load_feats(input_scp.at(nsample));

        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        la::cpu::tensor<double> input_tensor { la::cpu::vector<double> { raw_input } };

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        auto layers = make_fc(input, var_tree);

        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(layers.back());

        double max = -std::numeric_limits<double>::infinity();
        int argmax = -1;

        for (int i = 0; i < pred.vec_size(); ++i) {
            if (pred.data()[i] > max) {
                max = pred.data()[i];
                argmax = i;
            }
        }

        std::cout << input_scp.entries.at(nsample).key << std::endl;
        std::cout << id_label[argmax] << std::endl;
        std::cout << "." << std::endl;

        ++nsample;
    }
}

