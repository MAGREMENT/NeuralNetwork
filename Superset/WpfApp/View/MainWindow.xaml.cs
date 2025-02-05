using System.Windows;
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
    }

    public void SetDoodleData(double[,] data)
    {
        Drawer.SetData(data);
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