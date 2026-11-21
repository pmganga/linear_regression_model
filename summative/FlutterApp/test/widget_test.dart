import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:autism_diagnosis_age/main.dart';

void main() {
  testWidgets('Page has one labelled text field per model input, a Predict button and a result area',
      (WidgetTester tester) async {
    await tester.pumpWidget(const DiagnosisAgeApp());

    expect(find.byType(TextField), findsNWidgets(10));
    for (final input in allInputs) {
      expect(find.text(input.label), findsOneWidget);
    }
    expect(find.widgetWithText(FilledButton, 'Predict'), findsOneWidget);
    expect(find.byKey(const Key('result-area')), findsOneWidget);
  });
}
