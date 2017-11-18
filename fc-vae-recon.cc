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

struct vae_t {

    std::shared_ptr<autodiff::op_t> mu_e;
    std::shared_ptr<autodiff::op_t> log_sigma_e;

    std::vector<std::shared_ptr<autodiff::op_t>> hidden;

    std::shared_ptr<autodiff::op_t> z;
    std::shared_ptr<autodiff::op_t> mu_d;
    std::shared_ptr<autodiff::op_t> log_sigma_d;

};

vae_t make_vae(std::shared_ptr<autodiff::op_t> input,
    std::shared_ptr<tensor_tree::vertex> var_tree)
{
    vae_t vae;

    std::shared_ptr<autodiff::op_t> h = input;

    for (int i = 0; i < var_tree->children[0]->children.size() - 1; ++i) {
        auto a = autodiff::mul(h,
            tensor_tree::get_var(var_tree->children[0]->children[i]->children[0]));
        auto b = autodiff::rep_row_to(
            tensor_tree::get_var(var_tree->children[0]->children[i]->children[1]), a);

        h = autodiff::relu(autodiff::add(a, b));

        vae.hidden.push_back(h);
    }

    auto mu_e_a = autodiff::mul(vae.hidden.back(),
        tensor_tree::get_var(var_tree->children[0]->children.back()->children[0]));
    auto mu_e_b = autodiff::rep_row_to(
        tensor_tree::get_var(var_tree->children[0]->children.back()->children[1]),
        mu_e_a);
    vae.mu_e = autodiff::add(mu_e_a, mu_e_b);

    auto log_sigma_e_a = autodiff::mul(vae.hidden.back(),
        tensor_tree::get_var(var_tree->children[0]->children.back()->children[2]));
    auto log_sigma_e_b = autodiff::rep_row_to(
        tensor_tree::get_var(var_tree->children[0]->children.back()->children[3]),
        log_sigma_e_a);
    vae.log_sigma_e = autodiff::add(log_sigma_e_a, log_sigma_e_b);

    auto z = vae.mu_e;

    for (int i = 0; i < var_tree->children[1]->children.size() - 1; ++i) {
        auto a = autodiff::mul(z, tensor_tree::get_var(
            var_tree->children[1]->children[i]->children[0]));
        auto b = autodiff::rep_row_to(tensor_tree::get_var(
            var_tree->children[1]->children[i]->children[1]), a);

        h = autodiff::relu(autodiff::add(a, b));

        vae.hidden.push_back(h);
    }

    auto mu_d_a = autodiff::mul(vae.hidden.back(),
        tensor_tree::get_var(var_tree->children[1]->children.back()->children[0]));
    auto mu_d_b = autodiff::rep_row_to(
        tensor_tree::get_var(var_tree->children[1]->children.back()->children[1]),
        mu_d_a);
    vae.mu_d = autodiff::add(mu_d_a, mu_d_b);

    return vae;
}

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    tensor_tree::vertex root { "nil" };

    tensor_tree::vertex encoder { "nil" };

    for (int i = 0; i < layer - 1; ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("encoder weight"));
        ell.children.push_back(tensor_tree::make_tensor("encoder bias"));
        encoder.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }

    tensor_tree::vertex enc_mean_var { "nil" };
    enc_mean_var.children.push_back(tensor_tree::make_tensor("encoder mean weight"));
    enc_mean_var.children.push_back(tensor_tree::make_tensor("encoder mean bias"));
    enc_mean_var.children.push_back(tensor_tree::make_tensor("encoder variance weight"));
    enc_mean_var.children.push_back(tensor_tree::make_tensor("encoder variance bias"));
    encoder.children.push_back(std::make_shared<tensor_tree::vertex>(enc_mean_var));

    root.children.push_back(std::make_shared<tensor_tree::vertex>(encoder));

    tensor_tree::vertex decoder { "nil" };

    for (int i = 0; i < layer - 1; ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("decoder weight"));
        ell.children.push_back(tensor_tree::make_tensor("decoder bias"));
        decoder.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }

    tensor_tree::vertex dec_mean_var { "nil" };
    dec_mean_var.children.push_back(tensor_tree::make_tensor("decoder mean weight"));
    dec_mean_var.children.push_back(tensor_tree::make_tensor("decoder mean bias"));
    decoder.children.push_back(std::make_shared<tensor_tree::vertex>(dec_mean_var));

    root.children.push_back(std::make_shared<tensor_tree::vertex>(decoder));

    return std::make_shared<tensor_tree::vertex>(root);
}

struct prediction_env {

    batch::scp input_scp;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    int d_x;
    int d_z;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "fc-vae-recon",
        "Reconstruct with a VAE",
        {
            {"input-scp", "", true},
            {"param", "", true},
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

    d_x = tensor_tree::get_tensor(param->children[0]->children[0]->children[0]).size(0);
    d_z = tensor_tree::get_tensor(param->children[0]->children.back()->children[0]).size(0);
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

        vae_t vae = make_vae(input, var_tree);

        std::cout << input_scp.entries[nsample].key << std::endl;

        auto& recon = autodiff::get_output<la::cpu::tensor_like<double>>(vae.mu_d);

        for (int i = 0; i < recon.vec_size(); ++i) {
            if (i != 0) {
                std::cout << " ";
            }

            std::cout << recon.data()[i];
        }
        std::cout << std::endl;

        std::cout << "." << std::endl;

        ++nsample;
    }

}

