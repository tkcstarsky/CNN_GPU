#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <stdio.h>  
#include <sys/io.h>
#include "CNN.h"
   
using namespace std;

void buildCNN(NeuralNetwork& NN)
{
	NNLayer* pLayer;

	int ii, jj, kk;
	double initWeight;
	char label[100];
	int icNeurons = 0;

	// 第0层，输入层
	// 创建神经元，和输入的数量相等
	// 装有 29x29=841 像素的 vector，没有权值

	pLayer = new NNLayer("Layer00");
	NN.m_Layers.push_back(pLayer);

	for (ii = 0; ii < 841; ++ii)
	{
		sprintf(label, "Layer00_Neuron%04d_Num%06d", ii, icNeurons);
		pLayer->m_Neurons.push_back(new NNNeuron(label));
		icNeurons++;
	}

	// 第一层：
	// 是一个卷积层，有6个特征图，每个特征图大小为13x13，
	// 特征图中的每个单元是由5x5的卷积核从输入层卷积而成。
	// 因此，共有13x13x6 = 1014个神经元，(5x5+1)x6 = 156个权值

	pLayer = new NNLayer("Layer01", pLayer);
	NN.m_Layers.push_back(pLayer);

	for (ii = 0; ii < 1014; ++ii)
	{
		sprintf(label, "Layer00_Neuron%04d_Num%06d", ii, icNeurons);
		pLayer->m_Neurons.push_back(new NNNeuron(label));
		icNeurons++;
	}

	for (ii = 0; ii < 156; ++ii)
	{
		initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;  //均匀随机分布
		pLayer->m_Weights.push_back(new NNWeight(initWeight));
	}

	// 和前一层相连：这是难点
	// 前一层是位图，大小为29x29
	// 这层中的每个神经元都和特征图中的5x5卷积核相关，
	// 每次移动卷积核2个像素

	int kernelTemplate[25] = {
		0,  1,  2,  3,  4,
		29, 30, 31, 32, 33,
		58, 59, 60, 61, 62,
		87, 88, 89, 90, 91,
		116,117,118,119,120 };

	int iNumWeight;

	int fm;  // "fm" 代表 "feature map"

	for (fm = 0; fm < 6; ++fm)
	{
		for (ii = 0; ii < 13; ++ii)
		{
			for (jj = 0; jj < 13; ++jj)
			{
				// 26 是每个特征图的权值数量

				iNumWeight = fm * 26;
				NNNeuron& n = *(pLayer->m_Neurons[jj + ii * 13 + fm * 169]);

				n.AddConnection(ULONG_MAX, iNumWeight++);  // 偏移量

				for (kk = 0; kk < 25; ++kk)
				{
					// 注意：最大下标为840
					// 因为前一层中的神经元数量为841

					n.AddConnection(2 * jj + 58 * ii + kernelTemplate[kk], iNumWeight++);
				}
			}
		}
	}

	// 第二层：
	// 这层是卷积层，有50个特征图。每个特征图大小为5x5，
	// 特征图中的每个单元是由5x5的卷积核卷积前一层中的所有6个特征图而得，
	// 因此，有5x5x50 = 1250个神经元，(5x5+1)x6x50 = 7800个权值

	pLayer = new NNLayer("Layer02", pLayer);
	NN.m_Layers.push_back(pLayer);

	for (ii = 0; ii < 1250; ++ii)
	{
		sprintf(label, "Layer00_Neuron%04d_Num%06d", ii, icNeurons);
		pLayer->m_Neurons.push_back(new NNNeuron(label));
		icNeurons++;
	}

	for (ii = 0; ii < 7800; ++ii)
	{
		initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back(new NNWeight(initWeight));
	}

	// 和前一层相连：这是难点
	// 前一层的每个特征图都是大小为13x13的位图，共6个特征图.
	// 这层中的每个5x5特征图中的每个神经元都和6个5x5的卷积核相关。
	// 这层中的每个特征图都有6个不同的卷积核
	// 每次将特征图移动2个像素

	int kernelTemplate2[25] = {
		0,  1,  2,  3,  4,
		13, 14, 15, 16, 17,
		26, 27, 28, 29, 30,
		39, 40, 41, 42, 43,
		52, 53, 54, 55, 56 };


	for (fm = 0; fm < 50; ++fm)
	{
		for (ii = 0; ii < 5; ++ii)
		{
			for (jj = 0; jj < 5; ++jj)
			{
				// 26 是每个特征图的权值数
				iNumWeight = fm * 26;
				NNNeuron& n = *(pLayer->m_Neurons[jj + ii * 5 + fm * 25]);

				n.AddConnection(ULONG_MAX, iNumWeight++);  // bias weight

				for (kk = 0; kk < 25; ++kk)
				{
					// 注意：最大下标为1013
					// 因为前一层中有1014个神经元

					n.AddConnection(2 * jj + 26 * ii + kernelTemplate2[kk], iNumWeight++);
					n.AddConnection(169 + 2 * jj + 26 * ii + kernelTemplate2[kk], iNumWeight++);
					n.AddConnection(338 + 2 * jj + 26 * ii + kernelTemplate2[kk], iNumWeight++);
					n.AddConnection(507 + 2 * jj + 26 * ii + kernelTemplate2[kk], iNumWeight++);
					n.AddConnection(676 + 2 * jj + 26 * ii + kernelTemplate2[kk], iNumWeight++);
					n.AddConnection(845 + 2 * jj + 26 * ii + kernelTemplate2[kk], iNumWeight++);
				}
			}
		}
	}

	// 第3层：
	// 这是个全连接层，有100个单元
	// 由于是全连接层，这层中的每个单元都和前一层中所有的1250个单元相连
	// 因此，有100神经元，100*(1250+1) = 125100个权值

	pLayer = new NNLayer("Layer03", pLayer);
	NN.m_Layers.push_back(pLayer);

	for (ii = 0; ii < 100; ++ii)
	{
		sprintf(label, "Layer00_Neuron%04d_Num%06d", ii, icNeurons);
		pLayer->m_Neurons.push_back(new NNNeuron(label));
		icNeurons++;
	}

	for (ii = 0; ii < 125100; ++ii)
	{
		initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back(new NNWeight(initWeight));
	}

	// 和前一层相连：全连接

	iNumWeight = 0;  // 这层中，权值不共享

	for (fm = 0; fm < 100; ++fm)
	{
		NNNeuron& n = *(pLayer->m_Neurons[fm]);
		n.AddConnection(ULONG_MAX, iNumWeight++);  // 偏移量

		for (ii = 0; ii < 1250; ++ii)
		{
			n.AddConnection(ii, iNumWeight++);
		}
	}

	// 第4层，最后一层：
	// 这是个全连接层，有10个单元。
	// 由于是全连接层，每个神经元都和前一层中的所有100个神经元相连
	// 因此，有10个神经元，10*(100+1)=1010个权值

	pLayer = new NNLayer("Layer04", pLayer);
	NN.m_Layers.push_back(pLayer);

	for (ii = 0; ii < 10; ++ii)
	{
		sprintf(label, "Layer00_Neuron%04d_Num%06d", ii, icNeurons);
		pLayer->m_Neurons.push_back(new NNNeuron(label));
		icNeurons++;
	}

	for (ii = 0; ii < 1010; ++ii)
	{
		initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back(new NNWeight(initWeight));
	}

	// 和前一层相连：全连接

	iNumWeight = 0;  // 这层中的权值不共享

	for (fm = 0; fm < 10; ++fm)
	{
		NNNeuron& n = *(pLayer->m_Neurons[fm]);
		n.AddConnection(ULONG_MAX, iNumWeight++);  // 偏移量

		for (ii = 0; ii < 100; ++ii)
		{
			n.AddConnection(ii, iNumWeight++);
		}
	}
}

void img2vector(string filename, vector<double>& input)
{
	ifstream fin;
	char file[100];
	strcpy(file, (char*)filename.data());
	fin.open(file);
	char data[100];
	int d;
	for (int i = 0; i < 29; i++)
	{
		fin.getline(data, 100);
		for (int j = 0; j < 29; j++)
		{
			d = data[j] - '0';
			input.push_back((double)d);
		}
	}
	fin.close();
}

void getFiles(string path, vector<string>& files, vector<string>& files1)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;

	if ((hFile = _findfirst(p.assign(path).append("/*.txt").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//加入列表
			files.push_back(p.assign(path).append("/").append(fileinfo.name));
			files1.push_back(fileinfo.name);

		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}

int main()
{
	//获取文件
	int i, j, k;
	vector<string> file, file1;
	getFiles("./trainingDigits", file, file1);
	int n = file.size();  //训练文件夹下的文件数
	vector<int> label;
	char str[10];
	for (i = 0; i < n; i++)    //从文件名解析分类数字 
	{
		char* d = "_";
		char* str = (char*)file1[i].data();
		char* p = strtok(str, d);
		label.push_back(atoi(p));
	}

	NeuralNetwork CNN;
	buildCNN(CNN);         //构建CNN

	for (int epoch = 0; epoch < n; epoch++)     //对于每个训练文件
	{
		//cout << epoch << " ";
		CNN.m_input.clear();
		CNN.m_desiredOutput.clear();
		CNN.m_actualOutput.clear();
		for (i = 0; i < 10; i++)                //构造应输出向量
		{
			if (i == label[epoch])
				CNN.m_desiredOutput.push_back(1.0);
			else
				CNN.m_desiredOutput.push_back(-1.0);
		}
		img2vector(file[epoch], CNN.m_input);  //读取训练文件，并转为向量
		CNN.Calculate();
		CNN.Backpropagate();
	}

	//////////////////////测试/////////////////////////////
	file.clear();
	file1.clear();
	label.clear();
	getFiles("./testDigits", file, file1);
	int right = 0;
	n = file.size();    //测试文件数
	for (i = 0; i < n; i++)  //从文件名字中解析分类数字
	{
		char* d = "_";
		char* str = (char*)file[i].data();
		char* p = strtok(str, d);
		label.push_back(atoi(p));
	}
	cout << endl;
	for (j = 0; j < n; j++)        //对于每个测试文件
	{
		//cout << j << " ";
		CNN.m_input.clear();
		CNN.m_desiredOutput.clear();
		CNN.m_actualOutput.clear();
		img2vector(file[j], CNN.m_input); //读取测试文件，并转为向量
		CNN.Calculate();
		double max = CNN.m_actualOutput[0];
		int maxi = 0;
		for (i = 1; i < 10; i++)
		{
			if (CNN.m_actualOutput[i] > max)
			{
				max = CNN.m_actualOutput[i];
				maxi = i;
			}
		}
		if (maxi == label[j]) //如果结果正确
			right++;
	}
	cout << "正确个数：" << right << endl;
	cout << "总个数：" << n << endl;
	cout << "正确率：" << (double)right / (double)n << endl;  //正确率

	return 0;
}