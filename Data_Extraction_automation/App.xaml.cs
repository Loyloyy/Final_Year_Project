using System;
using System.IO;
using System.Diagnostics;
using System.Windows;
using System.Threading;
using System.Windows.Forms;
using Excel = Microsoft.Office.Interop.Excel;
using Clipboard = System.Windows.Clipboard;
using Action = System.Action;

namespace Data_Processing
{
    public partial class MainWindow : System.Windows.Window
    {
        string MPTfolderconvert; string temp1;
        string MPTfolderconverteda; string MPTfolderconverted1; string MPTfolderconverted2; string MPTfolderconverted3;

        static void Mainwindow(string[] args)
        {
            
        }

        private void MPTfolderselect_Click(object sender, RoutedEventArgs e)
        {
            var BB = new System.Windows.Forms.OpenFileDialog();
            if (BB.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                MPTfolderconvert = BB.FileName;
                System.IO.FileInfo File = new System.IO.FileInfo(BB.FileName);

                //OR

                System.IO.StreamReader reader = new System.IO.StreamReader(MPTfolderconvert);
                //etc

                MPTfoldername.Text = MPTfolderconvert;
                MPTfolderconverted1 = MPTfolderconvert.Substring(0, MPTfolderconvert.LastIndexOf("\\"));
                MPTfolderconverted2 = MPTfolderconverted1.Substring(0, MPTfolderconverted1.LastIndexOf("\\"));
                MPTfolderconverted3 = MPTfolderconverted2.Substring(0, MPTfolderconverted2.LastIndexOf("\\"));
                MPTfolderconverteda = MPTfolderconverted2.Substring(MPTfolderconverted2.LastIndexOf("\\") + 1);
            }
        }

        public void CreateFolder()
        {
            Directory.CreateDirectory("C:\\Users\\Alloy\\Desktop\\DataSet from VDE\\3.1 Cell Cycle EIS_as_of_27_Sept_2019\\" + MPTfolderconverteda + " (Excel)");
            //To create excel folder for converted files. Without this line, the creation of CSV file will have an error somehow
        }

        private void MPTtoxlsxstep_Click(object sender, RoutedEventArgs e)
        {
            MPTtoxlsxstep2();
        }

        public void MPTtoxlsxstep2()
        {
            selectMPTfolder.IsEnabled = false;
            MPTtoxlsxstep.IsEnabled = false;
            xlsxtodatastep.IsEnabled = false;
            everything.IsEnabled = false;
            matlabfolderselect.IsEnabled = false;
            lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting MPT to Excel. "));

            CreateFolder();
            //Console.WriteLine(MPTfolderconverted);
            DirectoryInfo d = new DirectoryInfo(MPTfolderconverted1);
            string[] allfiles = Directory.GetFiles(MPTfolderconverted2, "*.*", SearchOption.AllDirectories);
            FileInfo[] Files = d.GetFiles("*.mpt");
            foreach (string file in allfiles)
            {
                //string filezz = file.Name;
                //string excelname = filezz.Substring(0, filezz.LastIndexOf("."));

                if (file.Contains(".mpt"))
                {
                    lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting MPT to Excel. "));
                    string excelname1 = file.Substring(0, file.LastIndexOf("."));
                    string excelname2 = excelname1.Substring(excelname1.LastIndexOf("\\") + 1);

                    //Clipboard.SetText(File.ReadAllText(file));

                    var ExcelApp = new Excel.Application();
                    Excel.Workbook workbook = ExcelApp.Workbooks.Open(@"C:\Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\Excelling.xlsx");
                    ExcelApp.Visible = true;
                    lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting MPT to Excel.. "));
                    Clipboard.SetText(File.ReadAllText(file));
                    Thread.Sleep(2000);
                    SendKeys.SendWait("^{v}"); ;
                    Thread.Sleep(3000);
                    //Console.WriteLine(excelname);
                    Console.WriteLine(excelname2);
                    workbook.SaveAs(@"C:\Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\" + MPTfolderconverteda + " (Excel)\\" + excelname2, Microsoft.Office.Interop.Excel.XlFileFormat.xlWorkbookDefault);
                    workbook.Close();
                    ExcelApp.Quit();
                    lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting MPT to Excel... "));
                    Thread.Sleep(2000);
                }
                

            }
            selectMPTfolder.IsEnabled = true;
            MPTtoxlsxstep.IsEnabled = true;
            xlsxtodatastep.IsEnabled = true;
            everything.IsEnabled = true;
            matlabfolderselect.IsEnabled = true;
            lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Conversion to Excel finished! "));
        }

        private void matlabfolderselect_Click(object sender, RoutedEventArgs e) //Code under this area is for debugging purposes
        {
            Console.WriteLine(MPTfolderconverted1);
            Console.WriteLine(MPTfolderconverted2);
            Console.WriteLine(MPTfolderconverted3);
            Console.WriteLine(MPTfolderconverteda);

            string[] allfiles = Directory.GetFiles(MPTfolderconverted2, "*.*", SearchOption.AllDirectories);
        }

        private void everything_Click(object sender, RoutedEventArgs e)
        {
            MPTtoxlsxstep2();
            selectMPTfolder.IsEnabled = false;
            MPTtoxlsxstep.IsEnabled = false;
            xlsxtodatastep.IsEnabled = false;
            everything.IsEnabled = false;
            matlabfolderselect.IsEnabled = false;
            Thread.Sleep(15000);
            xlsxtodatastep2();
        }

        private void xlsxtodatastep_Click(object sender, RoutedEventArgs e)
        {
            selectMPTfolder.IsEnabled = false;
            MPTtoxlsxstep.IsEnabled = false;
            xlsxtodatastep.IsEnabled = false;
            everything.IsEnabled = false;
            matlabfolderselect.IsEnabled = false;
            xlsxtodatastep2();
            selectMPTfolder.IsEnabled = true;
            MPTtoxlsxstep.IsEnabled = true;
            xlsxtodatastep.IsEnabled = true;
            everything.IsEnabled = true;
            matlabfolderselect.IsEnabled = true;
        }

        public void xlsxtodatastep2()
        {
            lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting MPT to Excel. "));
            //add a line to check if the filesnames is already present, if it is, delete it
            if (File.Exists(@"C: \Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019" + "\\Filenames" + ".txt"))
            {
                System.IO.File.Delete(@"C:\Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\" + "Filenames.txt");
            }

            //Directory.CreateDirectory("C:\\Users\\Alloy\\Desktop\\DataSet from VDE\\3.1 Cell Cycle EIS_as_of_27_Sept_2019\\" + MPTfolderconverteda + " (Excel)");
            DirectoryInfo d = new DirectoryInfo(@"C:\Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\" + MPTfolderconverteda + " (Excel)");
            FileInfo[] Files = d.GetFiles("*.xlsx"); //Getting Text files
            using (FileStream filestream = new FileStream(@"C: \Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019" + "\\Filenames" + ".txt", FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
            using (var streamwriter = new StreamWriter(filestream))
            {
                streamwriter.AutoFlush = true;
                Console.SetOut(streamwriter);
                foreach (FileInfo file in Files)
                {
                    string firststep = file.Name;
                    string secondstep = firststep.Substring(0, firststep.LastIndexOf("."));
                    Console.WriteLine(secondstep);              
                }
            }

            string[] lines = File.ReadAllLines(@"C: \Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\Filenames.txt");
            foreach (string line in lines)
            {
                lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting Excel to Data. "));
                string fileName = "EISparameterestimation.m";
                string sourcePath = @"C:\Users\Alloy\Desktop";
                string targetPath = @"C:\Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\" + MPTfolderconverteda + " (Excel)";
                // Use Path class to manipulate file and directory paths.
                string sourceFile = System.IO.Path.Combine(sourcePath, fileName);
                string destFile = System.IO.Path.Combine(targetPath, fileName);
                System.IO.File.Copy(sourceFile, destFile, true);


                Process.Start(@"C:\Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\\" + MPTfolderconverteda + " (Excel)\\EISparameterestimation.m");
                Thread.Sleep(30000);
                lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting Excel to Data.. "));
                SendKeys.SendWait("^{f}");
                Thread.Sleep(1000);
                SendKeys.SendWait("S00418002054_1st Cycle_C01");
                SendKeys.SendWait("\t");
                Thread.Sleep(1000);
                SendKeys.SendWait(line);
                Thread.Sleep(1000);
                lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting Excel to Data... "));
                SendKeys.SendWait("\t");
                SendKeys.SendWait("\t");
                SendKeys.SendWait("\t");
                SendKeys.SendWait("\t");
                SendKeys.SendWait("\t");
                SendKeys.SendWait("\t");
                SendKeys.SendWait("\t");
                SendKeys.SendWait("\t");
                SendKeys.SendWait("{ENTER}");
                Thread.Sleep(500);
                SendKeys.SendWait("\t");
                SendKeys.SendWait("{ENTER}");

                Thread.Sleep(5000);
                SendKeys.SendWait("{F5}");
                Thread.Sleep(2000);
                lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting Excel to Data.... "));
                SendKeys.SendWait("{ENTER}");
                Thread.Sleep(20000);
                SendKeys.SendWait("^{q}");

                lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Converting Excel to Data..... "));
                System.IO.File.Delete(@"C:\Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\" + MPTfolderconverteda + " (Excel)\\EISparameterestimation.m");
                System.IO.File.Delete(@"C:\Users\Alloy\Desktop\DataSet from VDE\3.1 Cell Cycle EIS_as_of_27_Sept_2019\" + "Filenames.txt");
                Thread.Sleep(10000);
            }
            lblProgress.Dispatcher.BeginInvoke((Action)(() => lblProgress.Content = "Conversion to Data finished! "));
        }
    }

}
