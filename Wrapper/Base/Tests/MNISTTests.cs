using Base;

namespace Tests;

public class Tests
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
        const string? write = @"C:\Users\zacha\Downloads\";
        var data = MNIST.Read(
            @"mnist-data\t10k-labels.idx1-ubyte", 
            @"mnist-data\t10k-images.idx3-ubyte");

        var result = FlattenedData.FromGuessingPoints(data, 10);
        for (int i = 0; i < data.Count; i++)
        {
            Assert.That(result.Expected[i * 10 + data[i].Output], Is.EqualTo(1));
        }

        if (write is not null)
        {
            using var iWriter = File.Create(write + "images");
            foreach (var i in result.Inputs)
            {
                iWriter.Write(BitConverter.GetBytes(i));
            }
            
            using var eWriter = File.Create(write + "labels");
            foreach (var e in result.Expected)
            {
                eWriter.Write(BitConverter.GetBytes(e));
            }
        }
    }
}