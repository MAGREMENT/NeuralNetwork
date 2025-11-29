namespace Base;

public interface INeuralNetwork : IPointerWrapper, IDisposable
{
    public int Length { get; }
    
    public double[] Predict(double[] input);

    public double GetCost(FlattenedData data);
    
    public void Learn(FlattenedData data, int iterations, ILearningState? state);
}