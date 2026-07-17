from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Evaluator:

    @staticmethod
    def evaluate(y_true, y_pred):
        
        accuracy = accuracy_score(y_true, y_pred)

        matrix = confusion_matrix(y_true, y_pred)

        report = classification_report(y_true, y_pred)

        return accuracy, matrix, report