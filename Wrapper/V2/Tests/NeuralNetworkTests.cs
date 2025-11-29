using NeuralNetwork.V2;

namespace Tests;

public class Tests
{
    [Test]
    public void FromBuilderTest()
    {
        using var builder = Builder.FromYaml("/yaml-test.yaml");
        using var network = builder.Build();
        
        Assert.That(network.Length, Is.EqualTo(6)); //TODO fix (verify that yaml file is correctly found)
        Assert.That(network.InCount, Is.EqualTo(784));
        Assert.That(network.OutCount, Is.EqualTo(10));
    }
}