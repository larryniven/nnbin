#include "autodiff/autodiff.h"
#include "la/la.h"
#include "opt/opt.h"
#include <fstream>
#include <algorithm>
#include "nn/nn.h"

struct learning_env {
    std::ifstream input_list;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> labels;

    nn::param_t param;
    nn::opt_t opt_data;

    double step_size;

    std::string output_param;
    std::string output_opt_data;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();
};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn",
        "Train a feed-forward neural network",
        {
            {"input-list", "", true},
            {"label", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"output-param", "", true},
            {"output-opt-data", "", true}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    std::unordered_map<std::string, std::string> args = ebt::parse_args(argc, argv, spec);

    learning_env env { args };
    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    input_list.open(args.at("input-list"));

    std::tie(label_id, labels) = nn::load_label_map(args.at("label"));

    param = nn::load_param(args.at("param"));
    opt_data = nn::load_opt(args.at("opt-data"));

    step_size = std::stod(args.at("step-size"));

    output_param = args.at("output-param");
    output_opt_data = args.at("output-opt-data");
}

void learning_env::run()
{
    std::string line;

    int sample = 0;
    double accu_loss = 0;
    while (std::getline(input_list, line)) {
        std::vector<std::string> parts = ebt::split(line);

        std::vector<double> input;
        input.resize(parts.size() - 1);
        std::transform(parts.begin() + 1, parts.end(), input.begin(),
            [](std::string const& s) { return std::stod(s); });

        std::string label = parts.front();

        la::vector<double> gold;
        gold.resize(label_id.size());
        gold(label_id[label]) = 1;

        nn::nn_t nn = nn::make_nn(param);
        nn.hidden[0]->output = std::make_shared<la::vector<double>>(la::vector<double>(input));
        autodiff::eval(nn.output, autodiff::eval_funcs);
        nn::log_loss loss { autodiff::get_output<la::vector<double>>(nn.output), gold };

        accu_loss += loss.loss();

#if DEBUG
        {
            auto& tmp = param.weight[0](0, 0);
            double backup = tmp;
            tmp += 1e-8;

            nn::nn_t nn = nn::make_nn(param);

            nn.hidden[0]->output = std::make_shared<la::vector<double>>(la::vector<double>(input));
            autodiff::eval(nn.output, autodiff::eval_funcs);
            nn::log_loss loss2 { autodiff::get_output<la::vector<double>>(nn.output), gold };

            std::cout << "numerical grad: " << (loss2.loss() - loss.loss()) / 1e-8 << std::endl;
            tmp = backup;
        }
#endif

        nn.output->grad = std::make_shared<la::vector<double>>(loss.grad());
        autodiff::grad(nn.output, autodiff::grad_funcs);

#if DEBUG
        std::cout << "calc grad: " << autodiff::get_grad<la::matrix<double>>(nn.weight[0])(0, 0) << std::endl;
#endif

        nn::param_t grad = nn::copy_grad(nn);

        nn::adagrad_update(param, grad, opt_data, step_size);

        if (sample % 100 == 0) {
            std::cout << "last 100 avg loss: " << accu_loss / 100 << std::endl;
            accu_loss = 0;
        }

        ++sample;
    }

    save_param(param, output_param);
    save_opt(opt_data, output_opt_data);
}

