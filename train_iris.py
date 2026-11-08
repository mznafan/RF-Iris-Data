"""Contoh pelatihan klasifikasi dataset Iris dengan Random Forest."""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    # Muat fitur dan label dataset Iris bawaan scikit-learn.
    iris = load_iris()
    X, y = iris.data, iris.target

    # Sisihkan 20% data untuk pengujian dan pertahankan proporsi setiap kelas.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Buat dan latih model Random Forest.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi model pada data yang tidak digunakan saat pelatihan.
    y_pred = model.predict(X_test)
    print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
    print("\nLaporan klasifikasi:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=iris.target_names,
        )
    )
    print("Matriks kebingungan:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
