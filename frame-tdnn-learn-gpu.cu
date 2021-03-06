#include "la/la-gpu.h"
#include "autodiff/autodiff-gpu.h"
#include "nn/tensor-tree-gpu.h"
#include "nn/nn-gpu.h"
#include "ebt/ebt.h"
#include "util/speech.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include <vector>
#include "nn/lstm.h"
#include "nn/lstm-tensor-tree.h"
#include <random>
#include "nn/lstm-frame.h"
#include <algorithm>

struct tdnn_spec {
    std::vector<std::vector<int>> layers;
};

tdnn_spec read_spec(std::istream& is)
{
    std::vector<std::vector<int>> result;

    std::string line;

    while (std::getline(is, line) && line != "#") {
        std::vector<std::string> parts = ebt::split(line);

        std::vector<int> strides;

        for (int i = 0; i < parts.size(); ++i) {
            strides.push_back(std::stoi(parts[i]));
        }

        result.push_back(std::move(strides));
    }

    return tdnn_spec { result };
}

void write_spec(tdnn_spec const& spec, std::ostream& os)
{
    for (int i = 0; i < spec.layers.size(); ++i) {
        for (int j = 0; j < spec.layers[i].size(); ++j) {
            if (j != 0) {
                os << " ";
            }

            os << spec.layers[i][j];
        }
        os << std::endl;
    }
}

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
    tdnn_spec const& spec)
{
    tensor_tree::vertex root { "nil" };

    for (int i = 0; i < spec.layers.size(); ++i) {
        tensor_tree::vertex ell { "nil" };
        ell.children.push_back(tensor_tree::make_tensor("weight"));
        ell.children.push_back(tensor_tree::make_tensor("bias"));
        root.children.push_back(std::make_shared<tensor_tree::vertex>(ell));
    }

    tensor_tree::vertex softmax { "nil" };
    softmax.children.push_back(tensor_tree::make_tensor("softmax weight"));
    softmax.children.push_back(tensor_tree::make_tensor("softmax bias"));
    root.children.push_back(std::make_shared<tensor_tree::vertex>(softmax));

    return std::make_shared<tensor_tree::vertex>(root);
}

std::vector<std::shared_ptr<autodiff::op_t>>
split_frames(std::shared_ptr<autodiff::op_t> h,
    int nframes, unsigned int dim)
{
    std::vector<std::shared_ptr<autodiff::op_t>> result;

    unsigned int shift = 0;
    for (int t = 0; t < nframes; ++t) {
        result.push_back(autodiff::weak_var(h, shift, {dim}));
        shift += dim;
    }

    return result;
}

std::shared_ptr<autodiff::op_t> make_tdnn(
    tdnn_spec const& spec,
    std::shared_ptr<tensor_tree::vertex> var_tree,
    std::shared_ptr<autodiff::op_t> input,
    int nframes)
{
    std::shared_ptr<autodiff::op_t> feat = input;
    autodiff::computation_graph& graph = *(input->graph);

    for (int i = 0; i < spec.layers.size(); ++i) {

        auto b_param = tensor_tree::get_var(var_tree->children[i]->children[1]);
        unsigned int dim = autodiff::get_output<la::tensor_like<double>>(b_param).size(0);

        auto h = autodiff::mul(feat, tensor_tree::get_var(
            var_tree->children[i]->children[0]));

        std::vector<std::vector<std::shared_ptr<autodiff::op_t>>> hidden_vecs;

        for (int j = 0; j < spec.layers[i].size(); ++j) {
            std::vector<std::shared_ptr<autodiff::op_t>> vecs;

            for (int t = 0; t < nframes; ++t) {
                int shift = t * spec.layers[i].size() * dim + j * dim;
                vecs.push_back(weak_var(h, shift, {dim}));
            }

            hidden_vecs.push_back(vecs);
        }

        la::gpu::tensor<double> t;
        t.resize({(unsigned int) nframes, dim});

        auto storage = graph.var(t);
        std::vector<std::shared_ptr<autodiff::op_t>> storage_vecs
            = split_frames(storage, nframes, dim);

        std::vector<std::shared_ptr<autodiff::op_t>> outputs;

        for (int t = 0; t < nframes; ++t) {
            std::vector<std::shared_ptr<autodiff::op_t>> vecs;

            for (int j = 0; j < spec.layers[i].size(); ++j) {
                int k = spec.layers[i][j];

                if (t + k < 0 || t + k >= nframes) {
                    continue;
                }

                vecs.push_back(hidden_vecs[j][t+k]);
            }

            outputs.push_back(autodiff::add_to(storage_vecs[t], vecs));
        }

        auto b = autodiff::rep_row_to(b_param, storage);

        feat = autodiff::relu(autodiff::add(autodiff::weak_cat(outputs, storage), b));
    }

    auto z = autodiff::mul(feat, tensor_tree::get_var(
        var_tree->children.back()->children[0]));
    auto b = autodiff::rep_row_to(tensor_tree::get_var(
        var_tree->children.back()->children[1]), z);
    return autodiff::add(z, b);
}

struct learning_env {

    batch::scp frame_scp;
    batch::scp label_scp;

    tdnn_spec spec;
    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::string output_param;
    std::string output_opt_data;

    int seed;
    double dropout;
    double step_size;
    double clip;

    int batch_size;
    std::vector<int> indices;

    std::default_random_engine gen;

    std::unordered_map<std::string, int> label_id;

    std::unordered_set<std::string> ignored;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-tdnn-learn-gpu",
        "Train a TDNN frame classifier",
        {
            {"frame-scp", "", true},
            {"label-scp", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"label", "", true},
            {"ignore", "", false},
            {"dropout", "", false},
            {"shuffle", "", false},
            {"seed", "", false},
            {"batch-size", "", false},
            {"opt", "const-step,rmsprop,adagrad", true},
            {"step-size", "", true},
            {"clip", "", false},
            {"decay", "", false},
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

    learning_env env { args };

    env.run();

    std::cout << ebt::accu_timer<0>::msecs.count() / 1.0e6 / 60.0 << " min" << std::endl;

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_scp.open(args.at("frame-scp"));
    label_scp.open(args.at("label-scp"));

    std::ifstream param_ifs { args.at("param") };
    spec = read_spec(param_ifs);
    param = make_tensor_tree(spec);

    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    tensor_tree::gpu::to_device(param);

    step_size = std::stod(args.at("step-size"));

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    clip = std::numeric_limits<double>::infinity();
    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    std::vector<std::string> label_vec = util::load_label_set(args.at("label"));
    for (int i = 0; i < label_vec.size(); ++i) {
        label_id[label_vec[i]] = i;
    }

    if (ebt::in(std::string("ignore"), args)) {
        auto parts = ebt::split(args.at("ignore"), ",");
        ignored.insert(parts.begin(), parts.end());
    }

    dropout = 0;
    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    batch_size = 1;
    if (ebt::in(std::string("batch-size"), args)) {
        batch_size = std::stoi(args.at("batch-size"));
    }

    if (args.at("opt") == "rmsprop") {
        double decay = std::stod(args.at("decay"));
        opt = std::make_shared<tensor_tree::gpu::rmsprop_opt>(
            tensor_tree::gpu::rmsprop_opt(param, step_size, decay));
    } else if (args.at("opt") == "const-step") {
        opt = std::make_shared<tensor_tree::gpu::const_step_opt>(
            tensor_tree::gpu::const_step_opt(param, step_size));
    } else if (args.at("opt") == "adagrad") {
        opt = std::make_shared<tensor_tree::gpu::adagrad_opt>(
            tensor_tree::gpu::adagrad_opt(param, step_size));
    } else {
        throw std::logic_error("unknown optimizer " + args.at("opt"));
    }

    gen = std::default_random_engine { seed };

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    indices.resize(frame_scp.entries.size());

    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::shuffle(indices.begin(), indices.end(), gen);
    }
}

void learning_env::run()
{
    ebt::accu_timer<0> timer;

    int nsample = 0;

    std::shared_ptr<tensor_tree::vertex> accu_grad = tensor_tree::gpu::deep_copy(param);
    tensor_tree::gpu::zero(accu_grad);

    while (nsample < indices.size()) {
        std::cout << "sample: " << nsample << std::endl;
        std::cout << "index: " << indices[nsample] << std::endl;

        std::vector<std::vector<double>> frames = speech::load_frame_batch(
            frame_scp.at(indices[nsample]));

	std::vector<std::string> labels = speech::load_label_seq_batch(
            label_scp.at(indices[nsample]));

        std::cout << "frames: " << frames.size() << std::endl;
        std::cout << "labels: " << labels.size() << std::endl;

        assert(frames.size() == labels.size());

        autodiff::computation_graph graph;
        graph.eval_funcs = autodiff::gpu::eval_funcs;
        graph.grad_funcs = autodiff::gpu::grad_funcs;

        std::vector<double> input_cat;
        input_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            input_cat.insert(input_cat.end(), frames[i].begin(), frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::gpu::tensor<double>(la::cpu::weak_tensor<double>(input_cat.data(),
            { (unsigned int) frames.size(), (unsigned int) frames.front().size() })));

        input->grad_needed = false;

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::shared_ptr<autodiff::op_t> output = make_tdnn(spec, var_tree, input, frames.size());

        std::shared_ptr<autodiff::op_t> logprob = autodiff::logsoftmax(output);

        std::vector<double> gold_vec;
        gold_vec.resize(labels.size() * label_id.size());
        int nframes = 0;

        for (int t = 0; t < labels.size(); ++t) {
            if (!ebt::in(labels[t], ignored)) {
                gold_vec[t * label_id.size() + label_id.at(labels[t])] = 1;
                nframes += 1;
            }
        }

        la::gpu::tensor<double> gold { la::cpu::weak_tensor<double>(gold_vec.data(),
            {(unsigned int) labels.size(), (unsigned int) label_id.size()}) };
        auto& pred = autodiff::get_output<la::gpu::tensor_like<double>>(logprob);
        nn::gpu::log_loss loss { gold, pred };

        logprob->grad = std::make_shared<la::gpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            throw std::logic_error("loss is nan");
        }

        std::cout << "loss: " << ell / batch_size << std::endl;
        std::cout << "E: " << ell / nframes << std::endl;

        auto topo_order = autodiff::natural_topo_order(graph);

        autodiff::guarded_grad(topo_order, autodiff::gpu::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = tensor_tree::gpu::deep_copy(param);
        tensor_tree::gpu::zero(grad);

        tensor_tree::copy_grad(grad, var_tree);

        tensor_tree::gpu::axpy(accu_grad, 1, grad);

        if (nsample % batch_size == batch_size - 1) {

            tensor_tree::gpu::axpy(accu_grad, 1.0 / batch_size - 1, accu_grad);

            double n = tensor_tree::gpu::norm(accu_grad);

            std::cout << "grad norm: " << n << std::endl;

            if (ebt::in(std::string("clip"), args)) {
                if (n > clip) {
                    tensor_tree::gpu::axpy(accu_grad, clip / n - 1, accu_grad);
                    std::cout << "gradient clipped" << std::endl;
                }
            }

            std::vector<std::shared_ptr<tensor_tree::vertex>> vars
                = tensor_tree::leaves_pre_order(param);

            la::gpu::tensor<double> const& dv = tensor_tree::gpu::get_gpu_tensor(vars.front());

            la::cpu::tensor<double> v = la::gpu::to_host(dv);

            double v1 = v.data()[0];

            opt->update(accu_grad);

            tensor_tree::gpu::zero(accu_grad);

            v = la::gpu::to_host(dv);
            double v2 = v.data()[0];

            std::cout << "name: " << vars.front()->name << " weight: " << v1
                << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        }

        std::cout << std::endl;

        ++nsample;

#if DEBUG_TOP
        if (nsample >= DEBUG_TOP) {
            break;
        }
#endif
    }

    std::ofstream param_ofs { output_param };
    write_spec(spec, param_ofs);
    param_ofs << "#" << std::endl;
    tensor_tree::gpu::to_host(param);
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

