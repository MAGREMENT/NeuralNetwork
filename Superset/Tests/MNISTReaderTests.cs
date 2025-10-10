using Model;

namespace Tests;

public class MNISTReaderTests
{
    [Test]
    public void CountTest()
    {
        var result = MNIST.Read(
            @"mnist-data\t10k-labels.idx1-ubyte", 
            @"mnist-data\t10k-images.idx3-ubyte");
        
        Assert.That(result, Has.Count.EqualTo(10000));
    }

    [Test]
    public void FlattenTest()
    {
        var data = MNIST.Read(
            @"mnist-data\t10k-labels.idx1-ubyte", 
            @"mnist-data\t10k-images.idx3-ubyte");

        var result = FlattenedData.FromGuessingPoints(data, 10);
        for (int i = 0; i < data.Count; i++)
        {
            Assert.That(result.Expected[i * 10 + data[i].Output], Is.EqualTo(1));
        }
    }

    [Test]
    public void LearnTest()
    {
        using var n = new NeuralNetwork(784, 200, 100, 10);
        n.SetDataSelectorMiniBatch(64);
        n.SetThreadCount(4);
        n.SetActivationType(ActivationType.RELU, ActivationType.SOFTMAX);
        n.SetCostType(CostType.BINARY_CROSS_ENTROPY);
        n.SetLearningRate(0.05);
        n.InitializeWeightsAndBiases();
        
        var data = MNIST.Read(
            @"mnist-data\t10k-labels.idx1-ubyte", 
            @"mnist-data\t10k-images.idx3-ubyte", 10000);
        var flattened = FlattenedData.FromGuessingPoints(data, 10);
        Console.WriteLine(n.GetCost(flattened));
        Console.WriteLine(GuessingPoint.GetNetworkAccuracy(n, data));
        n.Learn(flattened, 20);
        Console.WriteLine(n.GetCost(flattened));
        Console.WriteLine(GuessingPoint.GetNetworkAccuracy(n, data));
        
        n.Save("/test.nn");
    }
}