#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/tensor-tree.h"
#include "nn/rsg.h"
#include "nn/nn.h"
#include <random>
#include <algorithm>

std::vector<std::shared_ptr<autodiff::op_t>> reconstruct(
    autodiff::computation_graph& comp_graph,
    std::vector<std::shared_ptr<autodiff::op_t>> const& seg_frames,
    std::shared_ptr<tensor_tree::vertex> var_tree,
    std::string const& label,
    std::unordered_map<std::string, int> const& label_id,
    int layer);

struct learning_env {

    std::ifstream frame_batch;

    std::shared_ptr<tensor_tree::vertex> param;

    int layer;

    std::vector<std::string> id_label;
    std::unordered_map<std::string, int> label_id;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> const& args);

    void run();

};

learning_env::learning_env(std::unordered_map<std::string, std::string> const& args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));

    std::string line;
    std::ifstream param_ifs { args.at("param") };
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = rsg::make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    id_label = speech::load_label_set(args.at("label"));
    for (int i = 0; i < id_label.size(); ++i) {
        label_id[id_label[i]] = i;
    }

}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "rsg-unsup-learn",
        "Train a Recurrent Sequence Generator",
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

    learning_env env { args };

    env.run();

    return 0;
}

void learning_env::run()
{
    std::unordered_map<std::string, std::pair<int, double>> accu_loss;

    int nsample = 0;

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;

        auto var_tree = tensor_tree::make_var_tree(comp_graph, param);

        std::vector<std::shared_ptr<autodiff::op_t>> seg_frames;

        for (int i = 0; i < frames.size() - 1; ++i) {
            seg_frames.push_back(comp_graph.var(
                la::tensor<double>(la::vector<double>(frames.at(i)))));
        }

        if (seg_frames.size() <= 1) {
            ++nsample;
            std::cout << std::endl;

            continue;
        }

        double inf = std::numeric_limits<double>::infinity();

        double min = inf;
        std::string argmin;

        for (auto& label: id_label) {
            std::vector<std::shared_ptr<autodiff::op_t>> outputs
                = reconstruct(comp_graph, seg_frames, var_tree, label, label_id, layer);

            double loss_sum = 0;

            for (int t = 0; t < outputs.size(); ++t) {
                la::tensor<double> gold { la::vector<double>(frames.at(t + 1)) };

                nn::l2_loss loss {
                    gold,
                    autodiff::get_output<la::tensor_like<double>>(outputs.at(t))
                };

                loss_sum += loss.loss();
            }

            if (loss_sum < min) {
                argmin = label;
                min = loss_sum;
            }
        }

        std::cout << nsample << ".label" << std::endl;

        std::cout << argmin << std::endl;

        std::cout << "." << std::endl;

        ++nsample;
    }
}

std::vector<std::shared_ptr<autodiff::op_t>> reconstruct(
    autodiff::computation_graph& comp_graph,
    std::vector<std::shared_ptr<autodiff::op_t>> const& seg_frames,
    std::shared_ptr<tensor_tree::vertex> var_tree,
    std::string const& label,
    std::unordered_map<std::string, int> const& label_id,
    int layer)
{
    lstm::lstm_multistep_transcriber multistep;

    for (int i = 0; i < layer; ++i) {
        multistep.steps.push_back(std::make_shared<lstm::dyer_lstm_step_transcriber>(
            lstm::dyer_lstm_step_transcriber{}));
    }

    std::vector<std::shared_ptr<autodiff::op_t>> outputs;

    la::vector<double> label_vec;
    label_vec.resize(label_id.size());
    label_vec(label_id.at(label)) = 1;

    auto label_embed = autodiff::mul(comp_graph.var(la::tensor<double>(label_vec)),
        tensor_tree::get_var(var_tree->children[0]));

    std::shared_ptr<autodiff::op_t> frame = seg_frames.front();

    for (int i = 0; i < seg_frames.size(); ++i) {
        la::vector<double> dur_vec;
        dur_vec.resize(100);
        dur_vec(seg_frames.size() - i) = 1;

        auto dur_embed = autodiff::mul(comp_graph.var(la::tensor<double>(dur_vec)),
            tensor_tree::get_var(var_tree->children[1]));
        auto acoustic_embed = autodiff::mul(seg_frames.at(i),
            tensor_tree::get_var(var_tree->children[2]));

        auto input_embed = autodiff::add(
            std::vector<std::shared_ptr<autodiff::op_t>>{ label_embed,
                dur_embed, acoustic_embed,
                tensor_tree::get_var(var_tree->children[3]) });

        auto output = multistep(var_tree->children[4], input_embed);

        outputs.push_back(autodiff::add(autodiff::mul(output,
            tensor_tree::get_var(var_tree->children[5])),
            tensor_tree::get_var(var_tree->children[6])));

        frame = outputs.back();
    }

    return outputs;
}
