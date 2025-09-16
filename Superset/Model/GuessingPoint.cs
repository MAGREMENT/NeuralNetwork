namespace Model;

public readonly struct GuessingPoint(double x, double y, int output)
{
    public readonly double X = x;
    public readonly double Y = y;
    public readonly int Output = output;

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
}

public delegate int GenerateOutput(double x, double y);

public readonly struct BoundingBox(double lowerX, double upperX, double lowerY, double upperY)
{
    public readonly double LowerX = lowerX;
    public readonly double UpperX = upperX;
    public readonly double LowerY = lowerY;
    public readonly double UpperY = upperY;
}