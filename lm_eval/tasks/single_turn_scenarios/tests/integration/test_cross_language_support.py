"""Integration tests for cross-language programming support."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sandbox import SandboxExecutor, LANGUAGE_CONFIGS
from metrics import syntax_validity, cyclomatic_complexity, code_style_score


class TestCrossLanguageSupport:
    """Integration tests for all supported programming languages."""
    
    @pytest.mark.integration
    def test_all_language_configs_exist(self):
        """Test that configurations exist for all supported languages."""
        expected_languages = ["python", "javascript", "java", "cpp", "go", "rust"]
        
        for language in expected_languages:
            assert language in LANGUAGE_CONFIGS, f"Missing config for {language}"
            
            config = LANGUAGE_CONFIGS[language]
            assert "image" in config, f"Missing Docker image for {language}"
            assert "extension" in config, f"Missing file extension for {language}"
            assert "run_cmd" in config, f"Missing run command for {language}"
    
    @pytest.mark.integration
    def test_python_language_support(self):
        """Test comprehensive Python language support."""
        python_code = """
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    print(fibonacci(10))

if __name__ == "__main__":
    main()
"""
        
        # Test syntax validation
        syntax_score = syntax_validity(python_code, "python")
        assert syntax_score == 1.0, "Valid Python code should have syntax score of 1.0"
        
        # Test cyclomatic complexity
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="3.0")
            complexity = cyclomatic_complexity(python_code, "python")
            assert complexity > 0, "Should calculate complexity for Python code"
        
        # Test code style
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            style_score = code_style_score(python_code, "python")
            assert style_score >= 0.0 and style_score <= 1.0, "Style score should be between 0 and 1"
        
        # Test sandbox execution (mocked)
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (0, b"55")  # fibonacci(10) = 55
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("python")
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        with patch('time.time', side_effect=[0, 1]):
                            result = executor.execute_code(python_code, [])
                            
                            assert result.exit_code == 0
                            assert "55" in result.stdout
    
    @pytest.mark.integration
    def test_javascript_language_support(self):
        """Test comprehensive JavaScript language support."""
        javascript_code = """
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

function main() {
    console.log(fibonacci(10));
}

main();
"""
        
        # Test syntax validation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(javascript_code, "javascript")
            assert syntax_score == 1.0, "Valid JavaScript code should have syntax score of 1.0"
        
        # Test invalid JavaScript syntax
        invalid_js = "function test( { console.log('invalid'); }"  # Missing closing parenthesis
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            syntax_score = syntax_validity(invalid_js, "javascript")
            assert syntax_score == 0.0, "Invalid JavaScript should have syntax score of 0.0"
        
        # Test sandbox execution (mocked)
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (0, b"55")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 1000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("javascript")
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        with patch('time.time', side_effect=[0, 1]):
                            result = executor.execute_code(javascript_code, [])
                            
                            assert result.exit_code == 0
    
    @pytest.mark.integration
    def test_java_language_support(self):
        """Test comprehensive Java language support."""
        java_code = """
public class Fibonacci {
    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
    
    public static void main(String[] args) {
        System.out.println(fibonacci(10));
    }
}
"""
        
        # Test syntax validation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(java_code, "java")
            assert syntax_score == 1.0, "Valid Java code should have syntax score of 1.0"
        
        # Test invalid Java syntax
        invalid_java = "public class Test { public static void main(String[] args { }"  # Missing closing parenthesis
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            syntax_score = syntax_validity(invalid_java, "java")
            assert syntax_score == 0.0, "Invalid Java should have syntax score of 0.0"
        
        # Test sandbox execution (mocked)
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (0, b"55")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 2 * 1024 * 1024},  # Java uses more memory
                    'cpu_stats': {'cpu_usage': {'total_usage': 2000000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("java")
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        with patch('time.time', side_effect=[0, 2]):  # Java compilation takes longer
                            result = executor.execute_code(java_code, [])
                            
                            assert result.exit_code == 0
    
    @pytest.mark.integration
    def test_cpp_language_support(self):
        """Test comprehensive C++ language support."""
        cpp_code = """
#include <iostream>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    std::cout << fibonacci(10) << std::endl;
    return 0;
}
"""
        
        # Test syntax validation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(cpp_code, "cpp")
            assert syntax_score == 1.0, "Valid C++ code should have syntax score of 1.0"
        
        # Test invalid C++ syntax
        invalid_cpp = "#include <iostream>\nint main( { return 0; }"  # Missing closing parenthesis
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            syntax_score = syntax_validity(invalid_cpp, "cpp")
            assert syntax_score == 0.0, "Invalid C++ should have syntax score of 0.0"
        
        # Test sandbox execution (mocked)
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (0, b"55")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 512 * 1024},  # C++ is memory efficient
                    'cpu_stats': {'cpu_usage': {'total_usage': 500000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("cpp")
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        with patch('time.time', side_effect=[0, 1.5]):
                            result = executor.execute_code(cpp_code, [])
                            
                            assert result.exit_code == 0
    
    @pytest.mark.integration
    def test_go_language_support(self):
        """Test comprehensive Go language support."""
        go_code = """
package main

import "fmt"

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

func main() {
    fmt.Println(fibonacci(10))
}
"""
        
        # Test syntax validation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(go_code, "go")
            assert syntax_score == 1.0, "Valid Go code should have syntax score of 1.0"
        
        # Test invalid Go syntax
        invalid_go = "package main\nfunc main( { }"  # Missing closing parenthesis
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            syntax_score = syntax_validity(invalid_go, "go")
            assert syntax_score == 0.0, "Invalid Go should have syntax score of 0.0"
        
        # Test sandbox execution (mocked)
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (0, b"55")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 1024 * 1024},
                    'cpu_stats': {'cpu_usage': {'total_usage': 800000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("go")
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        with patch('time.time', side_effect=[0, 1]):
                            result = executor.execute_code(go_code, [])
                            
                            assert result.exit_code == 0
    
    @pytest.mark.integration
    def test_rust_language_support(self):
        """Test comprehensive Rust language support."""
        rust_code = """
fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}

fn main() {
    println!("{}", fibonacci(10));
}
"""
        
        # Test syntax validation
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(rust_code, "rust")
            assert syntax_score == 1.0, "Valid Rust code should have syntax score of 1.0"
        
        # Test invalid Rust syntax
        invalid_rust = "fn main( { }"  # Missing closing parenthesis
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            syntax_score = syntax_validity(invalid_rust, "rust")
            assert syntax_score == 0.0, "Invalid Rust should have syntax score of 0.0"
        
        # Test sandbox execution (mocked)
        with patch('sandbox.DOCKER_AVAILABLE', True):
            with patch('docker.from_env') as mock_docker:
                mock_container = MagicMock()
                mock_container.exec_run.return_value = (0, b"55")
                mock_container.stats.return_value = iter([{
                    'memory_stats': {'max_usage': 512 * 1024},  # Rust is memory efficient
                    'cpu_stats': {'cpu_usage': {'total_usage': 600000}}
                }])
                
                mock_client = MagicMock()
                mock_client.containers.run.return_value = mock_container
                mock_docker.return_value = mock_client
                
                executor = SandboxExecutor("rust")
                
                with patch.object(executor, '_prepare_environment', return_value="/tmp/test"):
                    with patch.object(executor, '_cleanup'):
                        with patch('time.time', side_effect=[0, 2]):  # Rust compilation takes time
                            result = executor.execute_code(rust_code, [])
                            
                            assert result.exit_code == 0


class TestLanguageSpecificFeatures:
    """Integration tests for language-specific features and optimizations."""
    
    @pytest.mark.integration
    def test_python_specific_features(self):
        """Test Python-specific features like imports and libraries."""
        python_code_with_imports = """
import json
import math
from collections import defaultdict

def process_data():
    data = {"numbers": [1, 2, 3, 4, 5]}
    result = defaultdict(list)
    
    for num in data["numbers"]:
        result["squares"].append(math.pow(num, 2))
    
    return json.dumps(dict(result))

print(process_data())
"""
        
        # Test that Python-specific syntax is handled correctly
        syntax_score = syntax_validity(python_code_with_imports, "python")
        assert syntax_score == 1.0, "Python code with imports should be valid"
    
    @pytest.mark.integration
    def test_javascript_specific_features(self):
        """Test JavaScript-specific features like async/await and modules."""
        javascript_code_with_features = """
async function fetchData() {
    return new Promise((resolve) => {
        setTimeout(() => resolve([1, 2, 3, 4, 5]), 100);
    });
}

async function processData() {
    const data = await fetchData();
    const squares = data.map(num => num * num);
    console.log(JSON.stringify(squares));
}

processData();
"""
        
        # Test JavaScript async/await syntax
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(javascript_code_with_features, "javascript")
            assert syntax_score == 1.0, "JavaScript async code should be valid"
    
    @pytest.mark.integration
    def test_java_specific_features(self):
        """Test Java-specific features like generics and interfaces."""
        java_code_with_generics = """
import java.util.*;

interface Processor<T> {
    T process(T input);
}

public class DataProcessor implements Processor<List<Integer>> {
    @Override
    public List<Integer> process(List<Integer> input) {
        List<Integer> result = new ArrayList<>();
        for (Integer num : input) {
            result.add(num * num);
        }
        return result;
    }
    
    public static void main(String[] args) {
        DataProcessor processor = new DataProcessor();
        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
        List<Integer> squares = processor.process(data);
        System.out.println(squares);
    }
}
"""
        
        # Test Java generics and interfaces
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(java_code_with_generics, "java")
            assert syntax_score == 1.0, "Java code with generics should be valid"
    
    @pytest.mark.integration
    def test_cpp_specific_features(self):
        """Test C++-specific features like templates and STL."""
        cpp_code_with_templates = """
#include <iostream>
#include <vector>
#include <algorithm>

template<typename T>
class Processor {
public:
    std::vector<T> process(const std::vector<T>& input) {
        std::vector<T> result;
        std::transform(input.begin(), input.end(), std::back_inserter(result),
                      [](const T& x) { return x * x; });
        return result;
    }
};

int main() {
    Processor<int> processor;
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto squares = processor.process(data);
    
    for (const auto& square : squares) {
        std::cout << square << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
"""
        
        # Test C++ templates and STL
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(cpp_code_with_templates, "cpp")
            assert syntax_score == 1.0, "C++ code with templates should be valid"
    
    @pytest.mark.integration
    def test_go_specific_features(self):
        """Test Go-specific features like goroutines and channels."""
        go_code_with_goroutines = """
package main

import (
    "fmt"
    "sync"
)

func processNumbers(numbers []int, results chan<- int, wg *sync.WaitGroup) {
    defer wg.Done()
    for _, num := range numbers {
        results <- num * num
    }
}

func main() {
    numbers := []int{1, 2, 3, 4, 5}
    results := make(chan int, len(numbers))
    var wg sync.WaitGroup
    
    wg.Add(1)
    go processNumbers(numbers, results, &wg)
    
    wg.Wait()
    close(results)
    
    for square := range results {
        fmt.Print(square, " ")
    }
    fmt.Println()
}
"""
        
        # Test Go goroutines and channels
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(go_code_with_goroutines, "go")
            assert syntax_score == 1.0, "Go code with goroutines should be valid"
    
    @pytest.mark.integration
    def test_rust_specific_features(self):
        """Test Rust-specific features like ownership and traits."""
        rust_code_with_traits = """
trait Processor<T> {
    fn process(&self, input: Vec<T>) -> Vec<T>;
}

struct SquareProcessor;

impl Processor<i32> for SquareProcessor {
    fn process(&self, input: Vec<i32>) -> Vec<i32> {
        input.into_iter().map(|x| x * x).collect()
    }
}

fn main() {
    let processor = SquareProcessor;
    let data = vec![1, 2, 3, 4, 5];
    let squares = processor.process(data);
    
    for square in squares {
        print!("{} ", square);
    }
    println!();
}
"""
        
        # Test Rust traits and ownership
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            syntax_score = syntax_validity(rust_code_with_traits, "rust")
            assert syntax_score == 1.0, "Rust code with traits should be valid"


class TestCrossLanguageMetrics:
    """Integration tests for metrics calculation across languages."""
    
    @pytest.mark.integration
    def test_syntax_validation_consistency(self):
        """Test that syntax validation is consistent across languages."""
        # Valid code samples for each language
        valid_codes = {
            "python": "def hello(): print('Hello')",
            "javascript": "function hello() { console.log('Hello'); }",
            "java": "public class Hello { public static void main(String[] args) { System.out.println(\"Hello\"); } }",
            "cpp": "#include <iostream>\nint main() { std::cout << \"Hello\" << std::endl; return 0; }",
            "go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello\") }",
            "rust": "fn main() { println!(\"Hello\"); }"
        }
        
        # Invalid code samples (missing closing brace/parenthesis)
        invalid_codes = {
            "python": "def hello(\n    print('Hello')",
            "javascript": "function hello( { console.log('Hello'); }",
            "java": "public class Hello { public static void main(String[] args { }",
            "cpp": "#include <iostream>\nint main( { return 0; }",
            "go": "package main\nfunc main( { }",
            "rust": "fn main( { }"
        }
        
        for language in valid_codes:
            # Test valid code
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                valid_score = syntax_validity(valid_codes[language], language)
                assert valid_score == 1.0, f"Valid {language} code should score 1.0"
            
            # Test invalid code
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                invalid_score = syntax_validity(invalid_codes[language], language)
                assert invalid_score == 0.0, f"Invalid {language} code should score 0.0"
    
    @pytest.mark.integration
    def test_complexity_calculation_across_languages(self):
        """Test cyclomatic complexity calculation across languages."""
        # Simple function with if-else (complexity = 2)
        complexity_codes = {
            "python": "def test(x):\n    if x > 0:\n        return x\n    else:\n        return -x",
            "javascript": "function test(x) { if (x > 0) { return x; } else { return -x; } }",
            "java": "public static int test(int x) { if (x > 0) { return x; } else { return -x; } }",
            "cpp": "int test(int x) { if (x > 0) { return x; } else { return -x; } }",
            "go": "func test(x int) int { if x > 0 { return x } else { return -x } }",
            "rust": "fn test(x: i32) -> i32 { if x > 0 { x } else { -x } }"
        }
        
        for language, code in complexity_codes.items():
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="2.0")
                complexity = cyclomatic_complexity(code, language)
                assert complexity >= 1.0, f"{language} complexity should be at least 1.0"


class TestLanguageErrorHandling:
    """Integration tests for error handling across languages."""
    
    @pytest.mark.integration
    def test_compilation_error_handling(self):
        """Test handling of compilation errors across languages."""
        compilation_error_codes = {
            "java": "public class Test { public static void main(String[] args) { undefinedFunction(); } }",
            "cpp": "#include <iostream>\nint main() { undefinedFunction(); return 0; }",
            "go": "package main\nfunc main() { undefinedFunction() }",
            "rust": "fn main() { undefined_function(); }"
        }
        
        for language, code in compilation_error_codes.items():
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="compilation error")
                syntax_score = syntax_validity(code, language)
                # Should handle compilation errors gracefully
                assert syntax_score >= 0.0 and syntax_score <= 1.0
    
    @pytest.mark.integration
    def test_runtime_error_handling(self):
        """Test handling of runtime errors across languages."""
        runtime_error_codes = {
            "python": "def test(): return 1/0\ntest()",
            "javascript": "function test() { return 1/0; } test();",
            "java": "public class Test { public static void main(String[] args) { int x = 1/0; } }",
        }
        
        for language, code in runtime_error_codes.items():
            # Runtime errors should still have valid syntax
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)  # Syntax is valid
                syntax_score = syntax_validity(code, language)
                assert syntax_score == 1.0, f"{language} runtime error code should have valid syntax"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])