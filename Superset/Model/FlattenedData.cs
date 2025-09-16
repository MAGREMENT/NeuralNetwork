namespace Model;

public class FlattenedData(double[] inputs, int inputCutOff, double[] expected, int expectedCutOff)
{
    public readonly double[] Inputs = inputs;
    public readonly int InputCutOff = inputCutOff;
    public readonly double[] Expected = expected;
    public readonly int ExpectedCutOff = expectedCutOff;

    public int GetCount() => Inputs.Length / InputCutOff;

    public static FlattenedData FromArrays(IReadOnlyList<(double[], double[])> arrays)
    {
        var cutOff1 = arrays[0].Item1.Length;
        var cutOff2 = arrays[0].Item2.Length;

        var inputs = new double[arrays.Count * cutOff1];
        var expected = new double[arrays.Count * cutOff2];
        
        for (int i = 0; i < arrays.Count; i++)
        {
            var start = cutOff1 * i;
            for (int j = 0; j < cutOff1; j++)
            {
                inputs[start + j] = arrays[i].Item1[j];
            }
            
            start = cutOff2 * i;
            for (int j = 0; j < cutOff2; j++)
            {
                expected[start + j] = arrays[i].Item2[j];
            }
        }

        return new FlattenedData(inputs, cutOff1, expected, cutOff2);
    }

    public static FlattenedData FromGuessingPoints(IReadOnlyList<GuessingPoint> points, int outputValueCount)
    {
        var inputs = new double[points.Count * 2];
        var expected = new double[points.Count * outputValueCount];

        for (int i = 0; i < points.Count; i++)
        {
            var p = points[i];
            inputs[i * 2] = p.X;
            inputs[i * 2 + 1] = p.Y;

            var start = i * outputValueCount;
            for (int j = 0; j < outputValueCount; j++)
            {
                if (j == p.Output) expected[start + j] = 1;
            }
        }

        return new FlattenedData(inputs, 2, expected, outputValueCount);
    }
}