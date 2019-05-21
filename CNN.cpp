
#include "CNN.h"
#include "iostream"
using namespace std;

NeuralNetwork::NeuralNetwork()
{
	m_etaLearningRate = 0.001;               // 学习速率
}

NeuralNetwork::~NeuralNetwork()
{
	VectorLayers::iterator it;

	for (it = m_Layers.begin(); it < m_Layers.end(); it++)
	{
		delete* it;
	}
	m_Layers.clear();
	m_desiredOutput.clear();
	m_actualOutput.clear();
	m_input.clear();
}

//卷积神经网络中的正向传播，通过前一层计算每层的输出，得到最终输出向量
void NeuralNetwork::Calculate()
{
	VectorLayers::iterator lit = m_Layers.begin();
	VectorNeurons::iterator nit;

	// 第一层是输入层：直接输入
	// 根据输入向量，设置该层的输出

	if (lit < m_Layers.end())
	{
		nit = (*lit)->m_Neurons.begin();

		// 每个输入对应一个神经元
		int count = 0;
		while (nit != (*lit)->m_Neurons.end())
		{
			(*nit)->output = m_input[count];
			nit++;
			count++;
		}
	}

	// 通过Calculate()迭代剩余层

	for (lit++; lit < m_Layers.end(); lit++)
	{
		(*lit)->Calculate();
	}

	// 输出向量中存储最终结果

	lit = m_Layers.end();
	lit--;   //最后一层，输出层

	nit = (*lit)->m_Neurons.begin();

	while (nit != (*lit)->m_Neurons.end())
	{
		m_actualOutput.push_back((*nit)->output);
		nit++;
	}
}

//卷积神经网络中的误差反向传播
void NeuralNetwork::Backpropagate()
{
	/*
		误差反向传播，从最后一层一直到迭代到第一层
		Err：整个神经网络的输出误差
		Xn：第n层上的输出向量
		Xnm1：前一层的输出向量
		Wn：第n层的权值
		Yn：第n层的激活值
		F：激活函数 tanh
		F'：F的倒数 F'(Yn) = 1 - Xn^2
	*/

	VectorLayers::iterator lit = m_Layers.end() - 1;

	std::vector< double > dErr_wrt_dXlast((*lit)->m_Neurons.size()); //标准误差关于“输出值”的偏导
	std::vector< std::vector< double > > differentials;

	int iSize = m_Layers.size();

	differentials.resize(iSize);

	int ii;


	// differentials 存储标准误差 0.5*sumof( (actual-target)^2 ) 的偏导关于“输出值”的偏导

	for (ii = 0; ii < (*lit)->m_Neurons.size(); ++ii)
	{
		dErr_wrt_dXlast[ii] =
			m_actualOutput[ii] - m_desiredOutput[ii];  //实际输出 - 目标输出
	}


	// 存储dErr_wrt_dXlast
	// 为剩余需要存储在differentials中的vector预留空间，并初始化为0

	differentials[iSize - 1] = dErr_wrt_dXlast;  // 上一次

	for (ii = 0; ii < iSize - 1; ++ii)
	{
		differentials[ii].resize(
			m_Layers[ii]->m_Neurons.size(), 0.0);
	}

	/*
		迭代计算除了第一层之外的剩余层，每一层都要反向传播误差，
		并调整自己的权值
		用 differentials[ ii ] 计算 differentials[ ii - 1 ]
	*/

	ii = iSize - 1;
	for (lit; lit > m_Layers.begin(); lit--)
	{
		(*lit)->Backpropagate(differentials[ii],
			differentials[ii - 1], m_etaLearningRate);
		--ii;
	}

	differentials.clear();
}


////////////////////////////////////////////////////////////////////////////////////////////

NNLayer::NNLayer(char* str, NNLayer * pPrev)
{
	m_layerName = str;
	m_pPrevLayer = pPrev;
}

NNLayer::~NNLayer()
{
	VectorWeights::iterator wit;
	VectorNeurons::iterator nit;

	for (nit = m_Neurons.begin(); nit < m_Neurons.end(); nit++)
	{
		delete* nit;
	}

	for (wit = m_Weights.begin(); wit < m_Weights.end(); wit++)
	{
		delete* wit;
	}

	m_Weights.clear();
	m_Neurons.clear();
}

//每一层的计算，计算输出，正向传播
void NNLayer::Calculate()
{
	assert(m_pPrevLayer != NULL);

	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;

	double dSum;

	for (nit = m_Neurons.begin(); nit < m_Neurons.end(); nit++)  //该层的每个神经元
	{
		NNNeuron& n = *(*nit);

		cit = n.m_Connections.begin();

		assert((*cit).WeightIndex < m_Weights.size());

		// 第一个连接的权值是偏移量，忽略掉输入神经元的下标

		dSum = m_Weights[(*cit).WeightIndex]->value;

		for (cit++; cit < n.m_Connections.end(); cit++)   //每个连向当前神经元的连接
		{
			assert((*cit).WeightIndex < m_Weights.size());
			assert((*cit).NeuronIndex < m_pPrevLayer->m_Neurons.size());

			dSum += (m_Weights[(*cit).WeightIndex]->value) *
				(m_pPrevLayer->m_Neurons[(*cit).NeuronIndex]->output);
		}

		n.output = SIGMOID(dSum);   //当前神经元的输出

	}

}


//每一层的误差反向传播
void NNLayer::Backpropagate(std::vector< double > & dErr_wrt_dXn /* in */,
	std::vector< double > & dErr_wrt_dXnm1 /* out */,
	double etaLearningRate)
{
	/*
		Err：整个神经网络的输出误差
		Xn：第n层上的输出向量
		Xnm1：前一层的输出向量
		Wn：第n层的权值
		Yn：第n层的激活值
		F：激活函数 tanh
		F'：F的倒数 F'(Yn) = 1 - Xn^2
	*/

	int ii, jj, kk;
	int nIndex;
	double output;
	vector< double > dErr_wrt_dYn(m_Neurons.size());
	double* dErr_wrt_dWn = new double[m_Weights.size()];

	// 计算 : dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn，标准误差关于某个单元输入加权和的偏导

	for (ii = 0; ii < m_Neurons.size(); ++ii)
	{
		output = m_Neurons[ii]->output;

		dErr_wrt_dYn[ii] = DSIGMOID(output) * dErr_wrt_dXn[ii];
	}

	// 计算 : dErr_wrt_Wn = Xnm1 * dErr_wrt_Yn，标准误差关于权重的偏导
	// 对于这层中的每个神经元，通过前一层中与之相连的连接更新相关权值

	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;
	ii = 0;
	for (nit = m_Neurons.begin(); nit < m_Neurons.end(); nit++)
	{
		NNNeuron& n = *(*nit);

		for (cit = n.m_Connections.begin(); cit < n.m_Connections.end(); cit++)
		{
			kk = (*cit).NeuronIndex;
			if (kk == ULONG_MAX)
			{
				output = 1.0;  // 偏移量的权值
			}
			else
			{
				output = m_pPrevLayer->m_Neurons[kk]->output;
			}

			dErr_wrt_dWn[(*cit).WeightIndex] += dErr_wrt_dYn[ii] * output;
		}

		ii++;
	}


	// 计算 : dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn,
	// 计算这一层的每个神经元的dErr_wrt_dXnm1，
	// 作为下一层计算 dErr_wrt_dXn 的输入

	ii = 0;
	for (nit = m_Neurons.begin(); nit < m_Neurons.end(); nit++)
	{
		NNNeuron& n = *(*nit);

		for (cit = n.m_Connections.begin();
			cit < n.m_Connections.end(); cit++)
		{
			kk = (*cit).NeuronIndex;
			if (kk != ULONG_MAX)
			{
				// 不包括ULONG_MAX，意味着虚构的偏置神经元输出值永远都是1
				// 不训练偏置神经元

				nIndex = kk;

				dErr_wrt_dXnm1[nIndex] += dErr_wrt_dYn[ii] *
					m_Weights[(*cit).WeightIndex]->value;
			}

		}

		ii++;

	}


	// 计算 : 更新权重
	// 在这层中，使用了dErr_wrt_dW 和学习速率

	double oldValue, newValue;


	for (jj = 0; jj < m_Weights.size(); ++jj)
	{
		oldValue = m_Weights[jj]->value;
		newValue = oldValue - etaLearningRate * dErr_wrt_dWn[jj];
		m_Weights[jj]->value = newValue;
	}
}

/////////////////////////////////////////////////////////////////////////////////

NNNeuron::NNNeuron(char* str)
{
	m_neuronName = str;
	output = 0.0;
	m_Connections.clear();
}

NNNeuron::~NNNeuron()
{
	m_Connections.clear();
}


void NNNeuron::AddConnection(int iNeuron, int iWeight)
{
	m_Connections.push_back(NNConnection(iNeuron, iWeight));
}


void NNNeuron::AddConnection(NNConnection const& conn)
{
	m_Connections.push_back(conn);
}

//////////////////////////////////////////////////////////////////////////////

NNConnection::NNConnection(int iNeuron, int iWeight)
{
	NeuronIndex = iNeuron;
	WeightIndex = iWeight;
}

/////////////////////////////////////////////////////////////////////////////

NNWeight::NNWeight(double val)
{
	val = 0.0;
}