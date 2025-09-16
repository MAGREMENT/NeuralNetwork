using Model;

namespace WpfApp.Presenter;

public class GraphGuesserPresenter
{
    private readonly IGraphGuesserView _view;
    
    private readonly NeuralNetwork _network = new(2, 3, 2);
    private readonly List<GuessingPoint> _points = new();
    
    public int GenerateCount { get; set; } = 15;
    
    public BoundingBox Box { get; } = new(0, 50, 0, 50);

    public GraphGuesserPresenter(IGraphGuesserView view)
    {
        _view = view;
        _network.Randomize(0, 1);
    }

    public void Start()
    {
        _view.SetCost(_network.GetCost(FlattenedData.FromGuessingPoints(_points, 2)));
    }

    public void GeneratePoints()
    {
        _points.AddRange(GuessingPoint.GenerateRandom(GenerateParabolaOutput, Box, GenerateCount));
        _view.SetPoints(_points);
    }

    public void RemovePoints()
    {
        var v = Math.Min(_points.Count, GenerateCount);
        if (v == 0) return;

        _points.RemoveRange(_points.Count - v, v);
        _view.SetPoints(_points);
    }

    public int Predict(double x, double y)
    {
        return _network.Predict(new[] { x, y }).IndexOfHighestValue();
    }

    private static int GenerateSinusOutput(double x, double y)
    {
        return Math.Sin(x) * 10 > y ? 1 : 0;
    }

    private int GenerateParabolaOutput(double x, double y)
    {
        return y > -0.05 * x * x + 1.5 * x + Box.UpperY * 3 / 4 ? 1 : 0;
    }
}

public interface IGraphGuesserView
{
    public void SetPoints(IReadOnlyList<GuessingPoint> points);

    public void SetCost(double v);
}