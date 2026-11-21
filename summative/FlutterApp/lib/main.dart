import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// The API address on Render.
const String apiBaseUrl = 'https://autism-diagnosis-age-api.onrender.com';

void main() => runApp(const DiagnosisAgeApp());

/// One form field: its API name, label and help text.
class InputSpec {
  const InputSpec(this.key, this.label, this.help, {this.isNumber = false});

  final String key;
  final String label;
  final String help;
  final bool isNumber;
}

const List<InputSpec> childInputs = [
  InputSpec('severity', 'Autism severity', 'mild, moderate or severe'),
  InputSpec('adhd', 'ADHD diagnosis', 'yes or no'),
  InputSpec('anxiety', 'Anxiety diagnosis', 'yes or no'),
  InputSpec('developmental_delay', 'Developmental delay', 'yes or no'),
  InputSpec('speech_disorder', 'Speech or language disorder', 'yes or no'),
  InputSpec('intellectual_disability', 'Intellectual disability', 'yes or no'),
  InputSpec('sex', "Child's sex", 'male or female'),
];

const List<InputSpec> familyInputs = [
  InputSpec('income_poverty_ratio', 'Family income (% of poverty line)', 'number from 50 to 400', isNumber: true),
  InputSpec('insurance_type', 'Health insurance type', 'public, private, public and private, or uninsured'),
  InputSpec('home_language', 'Main language at home', 'english, spanish or other'),
];

const List<InputSpec> allInputs = [...childInputs, ...familyInputs];

class DiagnosisAgeApp extends StatelessWidget {
  const DiagnosisAgeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AutismCare',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final Map<String, TextEditingController> _controllers = {
    for (final input in allInputs) input.key: TextEditingController(),
  };

  bool _loading = false;
  String? _prediction; // e.g. "9.9 years"
  String? _predictionMessage;
  List<String> _errors = [];

  @override
  void dispose() {
    for (final controller in _controllers.values) {
      controller.dispose();
    }
    super.dispose();
  }

  /// Builds the request. Empty fields are skipped, so the API reports them as missing.
  Map<String, dynamic> _buildRequestBody() {
    final body = <String, dynamic>{};
    for (final input in allInputs) {
      final text = _controllers[input.key]!.text.trim();
      if (text.isEmpty) continue;
      if (input.isNumber) {
        body[input.key] = num.tryParse(text) ?? text;
      } else {
        body[input.key] = text.toLowerCase().replaceAll(' ', '_');
      }
    }
    return body;
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _loading = true;
      _prediction = null;
      _predictionMessage = null;
      _errors = [];
    });

    try {
      final response = await http
          .post(
            Uri.parse('$apiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(_buildRequestBody()),
          )
          .timeout(const Duration(seconds: 90)); // Render can take up to a minute to wake up

      final data = jsonDecode(response.body);
      if (response.statusCode == 200) {
        setState(() {
          _prediction = '${data['predicted_diagnosis_age_years']} years';
          _predictionMessage = data['message'] as String?;
        });
      } else if (response.statusCode == 422) {
        setState(() => _errors = _readValidationErrors(data['detail']));
      } else {
        setState(() => _errors = ['The server returned an error (code ${response.statusCode}). Please try again.']);
      }
    } on TimeoutException {
      setState(() => _errors = ['The server took too long to answer. It may be waking up, so please try again in a moment.']);
    } catch (_) {
      setState(() => _errors = ['Could not reach the server. Check your internet connection and try again.']);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  /// Turns the API's errors into short messages.
  List<String> _readValidationErrors(dynamic detail) {
    if (detail is! List) return ['Some values are not valid. Please check the form.'];
    final labels = {for (final input in allInputs) input.key: input.label};
    return detail.map<String>((error) {
      final location = (error['loc'] as List).last.toString();
      final label = labels[location] ?? location;
      switch (error['type']) {
        case 'missing':
          return '$label: this field is required.';
        case 'float_type':
        case 'float_parsing':
          return '$label: must be a number.';
        case 'string_type':
          return '$label: must be text.';
        default:
          return '$label: ${error['msg']}.';
      }
    }).toList();
  }

  Widget _buildField(InputSpec input) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextField(
        controller: _controllers[input.key],
        textInputAction: TextInputAction.next,
        decoration: InputDecoration(
          labelText: input.label,
          helperText: input.help,
          border: const OutlineInputBorder(),
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 12),
      child: Text(title, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
    );
  }

  Widget _buildResultArea() {
    final colors = Theme.of(context).colorScheme;
    Widget content;
    Color background = colors.surfaceContainerHighest;

    if (_loading) {
      content = const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2.5)),
          SizedBox(width: 12),
          Text('Predicting...'),
        ],
      );
    } else if (_errors.isNotEmpty) {
      background = colors.errorContainer;
      content = Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Please fix the following:',
              style: TextStyle(fontWeight: FontWeight.bold, color: colors.onErrorContainer)),
          const SizedBox(height: 8),
          for (final error in _errors)
            Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Text('• $error', style: TextStyle(color: colors.onErrorContainer)),
            ),
        ],
      );
    } else if (_prediction != null) {
      background = colors.primaryContainer;
      content = Column(
        children: [
          Text('Predicted age at autism diagnosis', style: TextStyle(color: colors.onPrimaryContainer)),
          const SizedBox(height: 6),
          Text(
            _prediction!,
            style: Theme.of(context)
                .textTheme
                .headlineMedium
                ?.copyWith(fontWeight: FontWeight.bold, color: colors.onPrimaryContainer),
          ),
          if (_predictionMessage != null) ...[
            const SizedBox(height: 6),
            Text(_predictionMessage!, textAlign: TextAlign.center, style: TextStyle(color: colors.onPrimaryContainer)),
          ],
        ],
      );
    } else {
      content = const Text('Fill in all 10 fields and tap Predict.', textAlign: TextAlign.center);
    }

    return Container(
      key: const Key('result-area'),
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: background, borderRadius: BorderRadius.circular(12)),
      child: content,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AutismCare: Diagnosis Age')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Estimate the age at which a child is likely to be diagnosed with autism, '
                'so families at risk of a late diagnosis can be connected to professionals sooner',
              ),
              const SizedBox(height: 12),
              _buildSectionTitle('About the child'),
              for (final input in childInputs) _buildField(input),
              _buildSectionTitle('About the family'),
              for (final input in familyInputs) _buildField(input),
              const SizedBox(height: 4),
              SizedBox(
                height: 48,
                child: FilledButton(
                  onPressed: _loading ? null : _predict,
                  child: const Text('Predict', style: TextStyle(fontSize: 16)),
                ),
              ),
              const SizedBox(height: 16),
              _buildResultArea(),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}
