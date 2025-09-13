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
            @"mnist-data\t10k-images.idx3-ubyte", 3);

        var result = MNIST.FlattenForNeuralNetwork(data);
        for (int i = 0; i < 3; i++)
        {
            Assert.That(result.Expected[i * 10 + data[i].Item1], Is.EqualTo(1));
        }
    }
}