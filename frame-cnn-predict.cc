#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "nn/lstm.h"
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/cnn.h"

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int conv_layer, int fc_layer)
{
    tensor_tree::vertex root { tensor_tree::tensor_t::nil };

    root.children.push_back(cnn::make_cnn_tensor_tree(conv_layer));

    for (int i = 0; i < fc_layer; ++i) {
        tensor_tree::vertex fc { tensor_tree::tensor_t::nil };
        fc.children.push_back(tensor_tree::make_tensor("softmax weight"));
        fc.children.push_back(tensor_tree::make_tensor("softmax bias"));
        root.children.push_back(std::make_shared<tensor_tree::vertex>(fc));
    }

    return std::make_shared<tensor_tree::vertex>(root);
}

struct prediction_env {

    std::ifstream frame_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    int conv_layer;
    int fc_layer;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-cnn-predict",
        "Predict with a CNN frame classifier",
        {
            {"frame-batch", "", true},
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
    frame_batch.open(args.at("frame-batch"));

    std::string line;

    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    std::vector<std::string> parts = ebt::split(line);
    conv_layer = std::stoi(parts[0]);
    fc_layer = std::stoi(parts[1]);
    param = make_tensor_tree(conv_layer, fc_layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 1;

    while (1) {
        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        std::cout << "sample: " << nsample << std::endl;

        autodiff::computation_graph graph;
        var_tree = tensor_tree::make_var_tree(graph, param);

        la::tensor<double> input_tensor;
        input_tensor.resize({ (unsigned int) frames.size(), (unsigned int) frames.front().size(), 1});

        for (int t = 0; t < frames.size(); ++t) {
            for (int i = 0; i < frames.front().size(); ++i) {
                input_tensor({t, i, 0}) = frames[t][i];
            }
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        cnn::multilayer_transcriber trans;

        for (int i = 0; i < conv_layer; ++i) {
            trans.layers.push_back(std::make_shared<cnn::cnn_transcriber>(
                cnn::cnn_transcriber{}));
        }

        std::shared_ptr<autodiff::op_t> feat = trans(input, var_tree->children[0]);

        auto& t = tensor_tree::get_tensor(param->children[0]->children.back()->children.back());

        feat = autodiff::reshape(feat,
            { (unsigned int) frames.size(), (unsigned int) frames.front().size() * t.size(0) });

        std::vector<std::shared_ptr<autodiff::op_t>> logprob;

        for (int t = 0; t < frames.size(); ++t) {
            std::shared_ptr<autodiff::op_t> feat_t = autodiff::row_at(feat, t);

            for (int i = 0; i < fc_layer - 1; ++i) {
                feat_t = autodiff::relu(autodiff::add(
                    autodiff::mul(feat_t, tensor_tree::get_var(var_tree->children[i + 1]->children[0])),
                    tensor_tree::get_var(var_tree->children[i + 1]->children[1])
                ));
            }

            logprob.push_back(autodiff::logsoftmax(
                autodiff::add(
                    autodiff::mul(feat_t,
                        tensor_tree::get_var(var_tree->children[fc_layer]->children[0])),
                    tensor_tree::get_var(var_tree->children[fc_layer]->children[1])
                )
            ));
        }

        double loss_sum = 0;
        double nframes = 0;

        // auto topo_order = autodiff::topo_order(logprob);
        // autodiff::eval(topo_order, autodiff::eval_funcs);

        for (int t = 0; t < frames.size(); ++t) {
            auto& pred = autodiff::get_output<la::tensor_like<double>>(logprob[t]);

            double max = -std::numeric_limits<double>::infinity();
            unsigned int argmax;

            for (int i = 0; i < pred.vec_size(); ++i) {
                if (pred({i}) > max) {
                    max = pred({i});
                    argmax = i;
                }
            }

            std::cout << id_label.at(argmax) << std::endl;
        }

        std::cout << "." << std::endl;

#if DEBUG_TOP
        if (nsample == DEBUG_TOP) {
            break;
        }
#endif

        ++nsample;
    }

}

