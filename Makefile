CXXFLAGS += -std=c++11 -I .. -L ../ebt -L ../la -L ../opt -L ../autodiff -L ../nn -L ../speech

.PHONY: all clean

bin = \
    learn \
    predict \
    loss-lstm \
    frame-lstm-learn \
    frame-lstm-predict \
    learn-gru \
    predict-gru \
    learn-residual \
    predict-residual \
    lstm-seg-ld-learn \
    lstm-seg-ld-predict \
    lstm-seg-li-learn \
    lstm-seg-li-predict \
    lstm-seg-li-avg \
    lstm-seg-li-grad \
    lstm-seg-li-update \
    lstm-seg-logp-learn \
    lstm-seg-logp-predict \
    rhn-learn \
    rhn-predict

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn: learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lebt -lblas

predict: predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lebt -lblas

frame-lstm-learn: frame-lstm-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

frame-lstm-predict: frame-lstm-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

loss-lstm: loss-lstm.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-gru: learn-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-gru: predict-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-residual: learn-residual.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-residual: predict-residual.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-ld-learn: lstm-seg-ld-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-ld-predict: lstm-seg-ld-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-learn: lstm-seg-li-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-predict: lstm-seg-li-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-avg: lstm-seg-li-avg.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-grad: lstm-seg-li-grad.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-update: lstm-seg-li-update.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-logp-learn: lstm-seg-logp-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-logp-predict: lstm-seg-logp-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rhn-learn: rhn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

rhn-predict: rhn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lspeech -lopt -lla -lebt -lblas

