using Model;

namespace WpfApp.Presenter;

public class GraphGuesserPresenter
{
    private readonly IGraphGuesserView _view;

    private readonly NeuralNetwork _network;
    private readonly List<GuessingPoint> _points = new();
    private readonly Dictionary<(double, double), int> _valueBuffer = new();
    
    public int GenerateCount { get; set; } = 15;
    
    public BoundingBox Box { get; } = new(0, 50, 0, 50);

    public GraphGuesserPresenter(IGraphGuesserView view)
    {
        _network = new(NeuralNetworkParameters.MomentumSigmoid, 2, 7, 2);
        _view = view;
        _network.Randomize(0, 1);
    }

    public void Start()
    {
        _view.SetCost(_network.GetCost(GetFlattenedData()));
    }

    public void GeneratePoints()
    {
        _points.AddRange(GuessingPoint.GenerateRandom(GenerateParabolaOutput, Box, GenerateCount));
        _view.SetPoints(_points);
        _view.SetCost(_network.GetCost(GetFlattenedData()));
    }

    public void RemovePoints()
    {
        var v = Math.Min(_points.Count, GenerateCount);
        if (v == 0) return;

        _points.RemoveRange(_points.Count - v, v);
        _view.SetPoints(_points);
        _view.SetCost(_network.GetCost(GetFlattenedData()));
    }

    public void Learn()
    {
        _network.Learn(GetFlattenedData(), _points.Count / 5 * 2, 50);
        
        _view.SetCost(_network.GetCost(GetFlattenedData()));
        _valueBuffer.Clear();
    }

    public int GetValueFor(double x, double y)
    {
        var entry = (x, y);
        if (!_valueBuffer.TryGetValue(entry, out var v))
        {
            v =  _network.Predict(new[] { x, y }).IndexOfHighestValue();
            _valueBuffer[entry] = v;
        }

        return v;
    }

    private FlattenedData GetFlattenedData()
    {
        return FlattenedData.FromGuessingPoints(_points, 2);
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