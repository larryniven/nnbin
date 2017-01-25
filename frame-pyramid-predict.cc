#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/lstm.h"
#include "nn/nn.h"
#include "nn/tensor-tree.h"
#include "nn/lstm-frame.h"

struct prediction_env {

    std::ifstream frame_batch;

    int layer;

    std::shared_ptr<tensor_tree::vertex> param;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::shared_ptr<tensor_tree::vertex> var_tree;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-pyramid-predict",
        "Predict frame labels with a pyramid LSTM",
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
    layer = std::stoi(line);
    param = lstm_frame::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

}

void prediction_env::run()
{
    int nsample = 0;

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph graph;
        std::vector<std::shared_ptr<autodiff::op_t>> inputs;

        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(graph.var(la::tensor<double>(la::vector<double>(frames[i]))));
        }

        var_tree = tensor_tree::make_var_tree(graph, param);

        std::shared_ptr<lstm::transcriber> trans
            = lstm_frame::make_pyramid_transcriber(layer, 0.0, nullptr);

        trans = std::make_shared<lstm::logsoftmax_transcriber>(
            lstm::logsoftmax_transcriber { trans });

        std::vector<std::shared_ptr<autodiff::op_t>> logprob = (*trans)(var_tree, inputs);

        double loss_sum = 0;
        double nframes = 0;

        auto topo_order = autodiff::topo_order(logprob);
        autodiff::eval(topo_order, autodiff::eval_funcs);

        std::cout << nsample << ".label" << std::endl;

        int freq = std::round(frames.size() / logprob.size());

        for (int t = 0; t < frames.size(); ++t) {
            auto& pred = autodiff::get_output<la::tensor_like<double>>(logprob[t / freq]);

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

