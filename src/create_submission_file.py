import datetime
import pandas as pd

def create_submission_file(model, X_train, y_train, X_test, save_name="model"):
    model.fit(X_train, y_train)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pd.DataFrame({
        "Id": X_test.index, 
        "Cover_Type": model.predict(X_test)
    }).to_csv(
        f"output/{save_name}_{timestamp}.csv", index=False
    )