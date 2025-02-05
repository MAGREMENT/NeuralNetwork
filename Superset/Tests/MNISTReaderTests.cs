using Model;

namespace Tests;

public class MNISTReaderTests
{
    [Test]
    public void CountTest()
    {
        var result = MNIST.Read(
            @"C:\Users\Zach\Desktop\Perso\NeuralNetwork\Superset\Model\mnist-data\t10k-labels.idx1-ubyte", 
            @"C:\Users\Zach\Desktop\Perso\NeuralNetwork\Superset\Model\mnist-data\t10k-images.idx3-ubyte");

        Console.WriteLine(result.Count);
        Assert.That(result, Has.Count.EqualTo(10000));
    }
}