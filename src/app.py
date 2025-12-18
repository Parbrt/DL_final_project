
def app():
    """
        cli app
    """
    print("---welcome to the cli app---\nDeep Learning project by TRAN Delvin, TCHAIWOU TCHEMTCHOUA Winnie Lena, ROBERT Paul-Aimé")
    flag = True
    while(flag):
        choice = input("\n\n1-load a simple model\n2-run a model optimizer\n3-exit")
        if choice == 1:
            choice = input("\n1-regression model\n2-classification model")
            if choice == 1:
                print("regression model on CS:GO data set, predicting the amount of money and hitpoint of the counter terrorist team using torch")
                
                from src.torch_impl.model_impl import RegModel
                model = RegModel()
                model.train()
                model.guess()
                
                choice = input("\n1-guess a random element\n2-get metrics")
                if choice == 1:
                    model.random_pred()
                    choice = 0
                elif choice == 2:
                    model.get_metrics()
                    choice = 0
                else:
                    print("wrong input")
                    choice = 0
            elif choice == 2:
                print("classification model on CS:GO data set, predicting the winning team using torch")
                
                from src.torch_impl.model_impl import CatModel
                model = CatModel()
                model.train()
                model.guess()
                
                choice = input("\n1-guess a random element\n2-get metrics")
                if choice == 1:
                    model.random_pred()
                    choice = 0
                elif choice == 2:
                    model.get_metrics()
                    choice = 0
                else:
                    print("wrong input")
                    choice = 0
            else:
                print("wrong input")
                choice = 0
        elif choice == 2:
            print("We optimized models by using the Optuna library\nThis library allows us to minimize the loss of the models by manipulating the hyperparametters")
            print("You can display the optimizations paramaters variations and performances on the sql app")
            choice = input("Which model would you like to optimize ?\n1-regression (tenserflow)\n2-classification (tenserflow)\n3-regression (torch)\n4-classification (torch)")
            if choice == 1:
                from src.main import run_regression_tf
                run_regression_tf()
                choice = 0
            elif choice == 2:
                from src.main import run_classification_tf
                run_classification_tf()
                choice = 0
            elif choice == 3:
                from src.main import run_regression_torch
                run_regression_torch()
                choice = 0
            elif choice == 4:
                from src.main import run_classification_torch
                run_classification_torch()
                choice = 0
            else:
                print("wrong input")
        elif choice == 3:
            print("exiting the app\nByebye")
            choice = 0
            flag = False
        else:
            print("wrong input")
