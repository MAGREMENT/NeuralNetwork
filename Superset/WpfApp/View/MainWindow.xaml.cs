using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using WpfApp.Presenter;

namespace WpfApp.View;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : IMainView
{
    private readonly MainPresenter _presenter;
    
    public MainWindow()
    {
        InitializeComponent();

        _presenter = new MainPresenter(this);
        Drawer.OnDraw += _presenter.Draw;
        Drawer.OnDrawStop += _presenter.Predict;
    }

    public void SetDoodleData(double[,] data)
    {
        Drawer.SetData(data);
    }

    public void SetPredictions(IReadOnlyList<(int, double)> predictions)
    {
        Predictions.Children.Clear();
        for (int i = 0; i < predictions.Count; i++)
        {
            var tb = new TextBlock
            {
                Padding = new Thickness(10),
                FontSize = 16,
                Foreground = i == 0 ? Brushes.Black : Brushes.DarkGray,
                Text = $"{predictions[i].Item1} - {predictions[i].Item2}%"
            };
            Predictions.Children.Add(tb);
        }
    }

    private void Previous(object sender, RoutedEventArgs e)
    {
        _presenter.Previous();
    }

    private void Next(object sender, RoutedEventArgs e)
    {
        _presenter.Next();
    }
}