// simplified view: some members have been omitted,
// and some signatures have been altered

// helpful typedef's

#include <vector>
#include <assert.h>
#include <math.h>
#include <climits>
using namespace std;

class NeuralNetwork;
class NNLayer;
class NNNeuron;
class NNConnection;
class NNWeight;

typedef std::vector< NNLayer* >  VectorLayers;
typedef std::vector< NNWeight* >  VectorWeights;
typedef std::vector< NNNeuron* >  VectorNeurons;
typedef std::vector< NNConnection > VectorConnections;

#define SIGMOID(x) (tanh(x))
#define DSIGMOID(x) (1-(SIGMOID(x))*(SIGMOID(x)))
#define UNIFORM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 )  //均匀随机分布

// 神经网络

class NeuralNetwork
{
public:
	NeuralNetwork();
	virtual ~NeuralNetwork();

	//正向传播，计算输出
	void Calculate();

	//反向传播，调整权值
	void Backpropagate();

	vector<double> m_input;            //输入向量
	vector<double> m_desiredOutput;    //理应输出的向量
	vector<double> m_actualOutput;     //实际输出的向量

	VectorLayers m_Layers;      //存储指向每一层的指针
	double m_etaLearningRate;   //学习速率
};


// 层

class NNLayer
{
public:
	NNLayer(char* str, NNLayer* pPrev = NULL);
	virtual ~NNLayer();

	//正向传播，计算输出
	void Calculate();

	//反向传播，调整权值
	void Backpropagate(std::vector< double >& dErr_wrt_dXn /* in */,
		std::vector< double >& dErr_wrt_dXnm1 /* out */,
		double etaLearningRate);

	char* m_layerName;         //该层的名称
	NNLayer* m_pPrevLayer;     //存储前一层的指针，以获得输入
	VectorNeurons m_Neurons;   //存储指向该层每个神经元的指针
	VectorWeights m_Weights;   //存储连向该层的每个连接的权值
};


// 神经元

class NNNeuron
{
public:
	NNNeuron(char* str);
	virtual ~NNNeuron();

	void AddConnection(int iNeuron, int iWeight);  //添加连接，(神经元下标, 权值下标)
	void AddConnection(NNConnection const& conn);

	char* m_neuronName;                //当前神经元的名称
	double output;                     //当前神经元的输出
	VectorConnections m_Connections;   //存储连向该神经元的所有连接，以获得该神经元的输入
};


// 连接

class NNConnection
{
public:
	NNConnection(int neuron = ULONG_MAX, int weight = ULONG_MAX);
	//virtual ~NNConnection();

	int NeuronIndex;         //神经元下标
	int WeightIndex;         //权值下标
};


// 权值

class NNWeight
{
public:
	NNWeight(double val = 0.0);
	//virtual ~NNWeight();

	double value;             //权值
};