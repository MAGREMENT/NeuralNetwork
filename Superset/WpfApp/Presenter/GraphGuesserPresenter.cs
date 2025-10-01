using Model;

namespace WpfApp.Presenter;

public class GraphGuesserPresenter : IDisposable 
{
    private readonly IGraphGuesserView _view;

    private const int batchSize = 32;
    private static readonly int[] _layers = { 2, 3, 2 };
    private readonly NeuralNetwork _network;
    private readonly LearningState _state;
    
    private readonly List<GuessingPoint> _points = new();
    private readonly Dictionary<(double, double), int> _valueBuffer = new();
    private bool _running = false;
    
    public int GenerateCount { get; set; } = 15;
    
    public BoundingBox Box { get; } = new(0, 50, 0, 50);

    public GraphGuesserPresenter(IGraphGuesserView view)
    {
        _network = new NeuralNetwork(NeuralNetworkParameters.NoMomentumSigmoid, _layers);
        _state = new LearningState(_network);
        _view = view;
    }

    public void Start()
    {
        _view.InitWeightsAndBiases(_layers);
        UpdateWeightsAndBiases();
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
        _running = true;
        Task.Run(() =>
        {
            while (_running)
            {
                _network.Learn(GetFlattenedData(), batchSize, 100, _state);
        
                _view.SetCost(_network.GetCost(GetFlattenedData()));
                UpdateWeightsAndBiases();
                _valueBuffer.Clear();
            }
        });
    }

    public void Stop()
    {
        _running = false;
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

    private void UpdateWeightsAndBiases()
    {
        for (int l = 0; l < _network.Length; l++)
        {
            var oc = _network.GetOutCount(l);
            for (int o = 0; o < oc; o++)
            {
                for (int i = 0; i < _network.GetInCount(l); i++)
                {
                    _view.SetWeight(l, i, o, oc, _network.GetWeight(l, i, o));
                }
                
                _view.SetBias(l, o, _network.GetBias(l, o));
            }
        }
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

    public void Dispose()
    {
        _network.Dispose();
    }
}

public interface IGraphGuesserView
{
    void InitWeightsAndBiases(int[] layers);
    void SetWeight(int layer, int input, int output, int outCount, double value);
    void SetBias(int layer, int output, double value);
    void SetPoints(IReadOnlyList<GuessingPoint> points);
    void SetCost(double v);
}