#include "la/la.h"
#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"
#include "nn/nn.h"
#include "ebt/ebt.h"
#include "util/speech.h"
#include "util/batch.h"
#include "util/util.h"
#include <fstream>
#include <vector>
#include <random>
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

        auto storage = autodiff::zeros(graph, {(unsigned int) nframes, dim});
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

    tdnn_spec spec;
    std::shared_ptr<tensor_tree::vertex> param;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-tdnn-predict",
        "Classify frames with a trained TDNN",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"label", "", true},
            {"print-logprob", "", false},
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

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_scp.open(args.at("frame-scp"));

    std::ifstream param_ifs { args.at("param") };
    spec = read_spec(param_ifs);
    param = make_tensor_tree(spec);

    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    id_label = util::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void learning_env::run()
{
    int nsample = 0;

    while (nsample < frame_scp.entries.size()) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(
            frame_scp.at(nsample));

        autodiff::computation_graph graph;

        std::vector<double> input_cat;
        input_cat.reserve(frames.size() * frames.front().size());

        for (int i = 0; i < frames.size(); ++i) {
            input_cat.insert(input_cat.end(), frames[i].begin(), frames[i].end());
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(
            la::cpu::weak_tensor<double>(input_cat.data(),
            { (unsigned int) frames.size(), (unsigned int) frames.front().size() }));

        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::shared_ptr<autodiff::op_t> output = make_tdnn(spec, var_tree, input, frames.size());

        std::shared_ptr<autodiff::op_t> logprob = autodiff::logsoftmax(output);

        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);

        std::cout << frame_scp.entries[nsample].key << std::endl;

        if (ebt::in(std::string("print-logprob"), args)) {
            for (int t = 0; t < pred.size(0); ++t) {
                for (int d = 0; d < pred.size(1); ++d) {
                    if (d != 0) {
                        std::cout << " ";
                    }

                    std::cout << pred({t, d});
                }

                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < pred.size(0); ++t) {
                double max = -std::numeric_limits<double>::infinity();
                int argmax = -1;

                for (int d = 0; d < pred.size(1); ++d) {
                    if (pred({t, d}) > max) {
                        max = pred({t, d});
                        argmax = d;
                    }
                }

                if (t != 0) {
                    std::cout << " ";
                }

                std::cout << id_label[argmax];
            }
            std::cout << std::endl;
        }

        std::cout << "." << std::endl;

        ++nsample;
    }
}

