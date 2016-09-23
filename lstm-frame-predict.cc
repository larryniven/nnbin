#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/lstm.h"
#include "nn/pred.h"
#include "nn/tensor_tree.h"
#include "nn/lstm-frame.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    lstm::stacked_bi_lstm_nn_t nn;
    rnn::pred_nn_t pred_nn;

    int layer;
    std::shared_ptr<tensor_tree::vertex> var_tree;

    std::vector<std::string> label;

    double dropout;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-lstm",
        "Predict frames with LSTM",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"label", "", true},
            {"dropout", "", false},
            {"logprob", "", false},
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

    std::ifstream param_ifs { args.at("param") };
    std::string line;
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = lstm_frame::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    label = speech::load_label_set(args.at("label"));

    if (ebt::in(std::string("dropout"), args)) {
        dropout = std::stod(args.at("dropout"));
    }
}

void prediction_env::run()
{
    int i = 1;

    while (1) {
        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph graph;
        std::vector<std::shared_ptr<autodiff::op_t>> inputs;

        for (int i = 0; i < frames.size(); ++i) {
            inputs.push_back(graph.var(la::vector<double>(frames[i])));
        }

        var_tree = tensor_tree::make_var_tree(graph, param);

        lstm::lstm_builder *builder;

        builder = new lstm::lstm_builder{};

        if (ebt::in(std::string("dropout"), args)) {
            nn = lstm::make_stacked_bi_lstm_nn_with_dropout(
                graph, var_tree->children[0], inputs, *builder, dropout);
        } else {
            nn = lstm::make_stacked_bi_lstm_nn(var_tree->children[0], inputs, *builder);
        }

        pred_nn = rnn::make_pred_nn(var_tree->children[1], nn.layer.back().output);

        auto topo_order = autodiff::topo_order(pred_nn.logprob);
        autodiff::eval(topo_order, autodiff::eval_funcs);

        std::cout << i << ".phn" << std::endl;

        if (ebt::in(std::string("logprob"), args)) {
            for (int t = 0; t < pred_nn.logprob.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob[t]);

                std::cout << pred(0);

                for (int j = 1; j < pred.size(); ++j) {
                    std::cout << " " << pred(j);
                }

                std::cout << std::endl;
            }
        } else {
            for (int t = 0; t < pred_nn.logprob.size(); ++t) {
                auto& pred = autodiff::get_output<la::vector<double>>(pred_nn.logprob[t]);

                int argmax = -1;
                double max = -std::numeric_limits<double>::infinity();

                for (int j = 0; j < pred.size(); ++j) {
                    if (pred(j) > max) {
                        max = pred(j);
                        argmax = j;
                    }
                }

                std::cout << label[argmax] << std::endl;
            }
        }

        std::cout << "." << std::endl;

        delete builder;

        ++i;
    }
}

