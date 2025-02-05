using Model;

namespace Tests;

public class NeuralNetworkTests
{
    [Test]
    public void LengthTest()
    {
        var network = new NeuralNetwork(new[] {2, 3, 2});
        Assert.That(network.Length, Is.EqualTo(2));

        network = new NeuralNetwork(new[] { 1, 1, 1, 1, 1 });
        Assert.That(network.Length, Is.EqualTo(4));
    }
}