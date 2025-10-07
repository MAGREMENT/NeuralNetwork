namespace Model;

public readonly struct GuessingPoint
{
    public readonly double[] Values;
    public readonly int Output;

    public GuessingPoint(double[] values, int output)
    {
        Values = values;
        Output = output;
    }

    public GuessingPoint(double x, double y, int output)
    {
        Values = new[] { x, y };
        Output = output;
    }

    public static IEnumerable<GuessingPoint> GenerateRandom(GenerateOutput generateFunc, BoundingBox box, int count)
    {
        var random = new Random();
        for (int i = 0; i < count; i++)
        {
            var x = random.NextDouble() * (box.UpperX - box.LowerX) + box.LowerX;
            var y = random.NextDouble() * (box.UpperY - box.LowerY) + box.LowerY;

            yield return new GuessingPoint(x, y, generateFunc(x, y));
        }
    }
    
    public static double GetNetworkAccuracy(NeuralNetwork network, IReadOnlyList<GuessingPoint> points)
    {
        var total = 0;
        foreach (var p in points)
        {
            if (network.Predict(p.Values).IndexOfHighestValue() == p.Output) total++;
        }

        return (double)total / points.Count * 100;
    }
}

public delegate int GenerateOutput(double x, double y);

public readonly struct BoundingBox(double lowerX, double upperX, double lowerY, double upperY)
{
    public readonly double LowerX = lowerX;
    public readonly double UpperX = upperX;
    public readonly double LowerY = lowerY;
    public readonly double UpperY = upperY;
}