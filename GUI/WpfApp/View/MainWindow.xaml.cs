using System.Windows.Navigation;

namespace WpfApp.View;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow
{
    private readonly bool _cancelNavigation = false;

    public MainWindow()
    {
        InitializeComponent();

        Frame.Content = new DoodleGuesser();
        _cancelNavigation = true;
    }

    private void CancelNavigation(object sender, NavigatingCancelEventArgs e)
    {
        if(_cancelNavigation) e.Cancel = true;
    }
}