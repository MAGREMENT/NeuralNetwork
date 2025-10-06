using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using Model;
using WpfApp.Presenter;

namespace WpfApp.View;

public partial class GraphGuesser : IGraphGuesserView
{
    private readonly GraphGuesserPresenter _presenter;
    private StackPanel[] _weigths;
    private StackPanel[] _biases;
    
    public GraphGuesser()
    {
        InitializeComponent();

        _presenter = new GraphGuesserPresenter(this);
        Graph.OutputGetter = _presenter.GetValueFor;
        
        _presenter.Start();
    }

    public void InitWeightsAndBiases(int[] layers)
    {
        _weigths = new StackPanel[layers.Length];
        _biases = new StackPanel[layers.Length];
        
        for (int i = 0; i < layers.Length - 1; i++)
        {
            var w = new StackPanel
            {
                Orientation = Orientation.Vertical,
                Margin = new Thickness(0, 20, 0, 0)
            };

            w.Children.Add(new TextBlock
            {
                Text = "Weights " + (i + 1)
            });

            var count = layers[i] * layers[i + 1];
            for (int j = 0; j < count; j++)
            {
                w.Children.Add(new Slider
                {
                    Minimum = -25,
                    Maximum = 25
                });
            }

            ControlPanel.Children.Add(w);
            _weigths[i] = w;
            
            var b = new StackPanel
            {
                Orientation = Orientation.Vertical,
                Margin = new Thickness(0, 20, 0, 0)
            };

            b.Children.Add(new TextBlock
            {
                Text = "Biases " + (i + 1)
            });
            
            for (int j = 0; j < layers[i + 1]; j++)
            {
                b.Children.Add(new Slider
                {
                    Minimum = -25,
                    Maximum = 25
                });
            }
            
            ControlPanel.Children.Add(b);
            _biases[i] = b;
        }
    }

    public void SetWeight(int layer, int input, int output, int outCount, double value)
    {
        Dispatcher.Invoke(() =>
        {
            var panel = _weigths[layer];
            ((Slider)panel.Children[1 + input * outCount + output]).Value = value;
        });
    }

    public void SetBias(int layer, int output, double value)
    {
        Dispatcher.Invoke(() =>
        {
            ((Slider)_biases[layer].Children[1 + output]).Value = value;
        });
    }

    public void Generate(object? o, RoutedEventArgs args)
    {
        Graph.SetMaxValues(_presenter.Box.UpperX, _presenter.Box.UpperY);
        _presenter.GeneratePoints();
    }

    public void Remove(object? o, RoutedEventArgs args)
    {
        _presenter.RemovePoints();
    }

    public void SetPoints(IReadOnlyList<GuessingPoint> points)
    {
        Graph.Points = points;
    }

    public void SetCost(double v)
    {
        Dispatcher.Invoke(() => CostBlock.Text = v.ToString("0.000000", CultureInfo.CreateSpecificCulture("en-US")));
    }

    private void Learn(object sender, RoutedEventArgs e)
    {
        _presenter.Learn();
        Graph.Refresh();
    }


    private void Stop(object sender, RoutedEventArgs e)
    {
        _presenter.Stop();
    }
}