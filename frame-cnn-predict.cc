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
#include "nn/cnn-frame.h"

struct prediction_env {

    std::ifstream frame_batch;

    cnn::cnn_t cnn_config;

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
    cnn_config = cnn::load_param(param_ifs);
    param = cnn_config.param;
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

        la::cpu::tensor<double> input_tensor;
        input_tensor.resize({ (unsigned int) frames.size(), 40, 3});

        for (int t = 0; t < frames.size(); ++t) {
            for (int i = 0; i < frames.front().size(); ++i) {
                input_tensor({t, i % 40, i / 40}) = frames[t][i];
            }
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        std::shared_ptr<cnn::transcriber> trans = cnn::make_transcriber(cnn_config, 0.0, nullptr);
        std::shared_ptr<autodiff::op_t> hidden = (*trans)(var_tree, input);

        trans = std::make_shared<cnn::logsoftmax_transcriber>(
            cnn::logsoftmax_transcriber{});

        std::shared_ptr<autodiff::op_t> logprob = (*trans)(var_tree->children.back(), hidden);

        auto& pred = autodiff::get_output<la::cpu::tensor_like<double>>(logprob);

        for (int t = 0; t < frames.size(); ++t) {
            double max = -std::numeric_limits<double>::infinity();
            unsigned int argmax;

            for (int i = 0; i < pred.size(1); ++i) {
                if (pred({t, i}) > max) {
                    max = pred({t, i});
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

