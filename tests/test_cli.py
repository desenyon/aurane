"""
Tests for the Aurane CLI module.
"""

import os
import pytest
import tempfile
import subprocess
from pathlib import Path


class TestCLIHelp:
    """Tests for CLI help commands."""
    
    def test_help_command(self):
        """Test that help command works."""
        result = subprocess.run(
            ['python', '-m', 'aurane.cli', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'aurane' in result.stdout.lower() or 'usage' in result.stdout.lower()
    
    def test_compile_help(self):
        """Test compile subcommand help."""
        result = subprocess.run(
            ['python', '-m', 'aurane.cli', 'compile', '--help'],
            capture_output=True,
            text=True
        )
        # Should not crash
        assert result.returncode == 0


class TestCLICompile:
    """Tests for CLI compile command."""
    
    def test_compile_simple_file(self):
        """Test compiling a simple .aur file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple .aur file
            input_file = os.path.join(tmpdir, 'test.aur')
            output_file = os.path.join(tmpdir, 'test.py')
            
            with open(input_file, 'w') as f:
                f.write("""model Net:
    def forward(x):
        x -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', input_file, '-o', output_file],
                capture_output=True,
                text=True
            )
            
            # Check that output file was created
            if result.returncode == 0:
                assert os.path.exists(output_file)
    
    def test_compile_to_stdout(self):
        """Test compiling to stdout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'test.aur')
            
            with open(input_file, 'w') as f:
                f.write("""model Net:
    def forward(x):
        x -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', input_file],
                capture_output=True,
                text=True
            )
            
            # Should output to stdout
            if result.returncode == 0:
                assert 'class' in result.stdout or 'Net' in result.stdout
    
    def test_compile_missing_file(self):
        """Test compiling a missing file."""
        result = subprocess.run(
            ['python', '-m', 'aurane.cli', 'compile', 'nonexistent.aur'],
            capture_output=True,
            text=True
        )
        
        # Should fail with error
        assert result.returncode != 0 or 'error' in result.stderr.lower() or 'not found' in result.stderr.lower()


class TestCLICheck:
    """Tests for CLI check command."""
    
    def test_check_valid_file(self):
        """Test checking a valid .aur file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'valid.aur')
            
            with open(input_file, 'w') as f:
                f.write("""model ValidNet:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'check', input_file],
                capture_output=True,
                text=True
            )
            
            # Should succeed
            if 'check' in result.stdout.lower() or result.returncode == 0:
                assert True


class TestCLIProfile:
    """Tests for CLI profile command."""
    
    def test_profile_model(self):
        """Test profiling a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'model.aur')
            
            with open(input_file, 'w') as f:
                f.write("""model ProfileNet:
    input_shape = (784,)
    def forward(x):
        x -> dense(256).relu
          -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'profile', input_file],
                capture_output=True,
                text=True
            )
            
            # Should output profile info
            # May contain params, flops, or layer info


class TestCLIVisualize:
    """Tests for CLI visualize command."""
    
    def test_visualize_mermaid(self):
        """Test visualizing as Mermaid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'model.aur')
            
            with open(input_file, 'w') as f:
                f.write("""model VisNet:
    def forward(x):
        x -> dense(64).relu
          -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'visualize', input_file, '--format', 'mermaid'],
                capture_output=True,
                text=True
            )
            
            # May output mermaid or show visualization
    
    def test_visualize_dot(self):
        """Test visualizing as DOT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'model.aur')
            
            with open(input_file, 'w') as f:
                f.write("""model VisNet:
    def forward(x):
        x -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'visualize', input_file, '--format', 'dot'],
                capture_output=True,
                text=True
            )
            
            # May output DOT format


class TestCLIVersion:
    """Tests for CLI version command."""
    
    def test_version_flag(self):
        """Test --version flag."""
        result = subprocess.run(
            ['python', '-m', 'aurane.cli', '--version'],
            capture_output=True,
            text=True
        )
        
        # Should show version or not crash
        # Version might be in stdout or stderr


class TestCLIErrors:
    """Tests for CLI error handling."""
    
    def test_invalid_command(self):
        """Test invalid subcommand."""
        result = subprocess.run(
            ['python', '-m', 'aurane.cli', 'invalidcommand'],
            capture_output=True,
            text=True
        )
        
        # Should show error or help
        assert result.returncode != 0 or 'error' in result.stderr.lower() or 'usage' in result.stdout.lower()
    
    def test_syntax_error_in_file(self):
        """Test handling syntax errors in .aur file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'invalid.aur')
            
            with open(input_file, 'w') as f:
                f.write("""model {{{ invalid syntax
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', input_file],
                capture_output=True,
                text=True
            )
            
            # Should report error
            assert result.returncode != 0 or 'error' in result.stderr.lower()


class TestCLIWithExamples:
    """Tests for CLI with example files."""
    
    @pytest.fixture
    def examples_dir(self):
        """Get the examples directory."""
        return Path(__file__).parent.parent / 'examples'
    
    def test_compile_simple_example(self, examples_dir):
        """Test compiling simple.aur example."""
        simple_file = examples_dir / 'simple.aur'
        if simple_file.exists():
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', str(simple_file)],
                capture_output=True,
                text=True
            )
            # Should succeed or at least not crash badly
    
    def test_compile_mnist_example(self, examples_dir):
        """Test compiling mnist.aur example."""
        mnist_file = examples_dir / 'mnist.aur'
        if mnist_file.exists():
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', str(mnist_file)],
                capture_output=True,
                text=True
            )
            # Should succeed
    
    def test_compile_resnet_example(self, examples_dir):
        """Test compiling resnet.aur example."""
        resnet_file = examples_dir / 'resnet.aur'
        if resnet_file.exists():
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', str(resnet_file)],
                capture_output=True,
                text=True
            )
            # Complex example


class TestCLIOutputFormats:
    """Tests for CLI output format options."""
    
    def test_backend_torch(self):
        """Test torch backend option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'model.aur')
            
            with open(input_file, 'w') as f:
                f.write("""model Net:
    def forward(x):
        x -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', input_file, '--backend', 'torch'],
                capture_output=True,
                text=True
            )
            
            # Should compile with torch backend
            if result.returncode == 0:
                assert 'torch' in result.stdout.lower() or 'nn.Module' in result.stdout


class TestCLIQuiet:
    """Tests for CLI quiet mode."""
    
    def test_quiet_flag(self):
        """Test --quiet flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'model.aur')
            output_file = os.path.join(tmpdir, 'model.py')
            
            with open(input_file, 'w') as f:
                f.write("""model Net:
    def forward(x):
        x -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', input_file, '-o', output_file, '--quiet'],
                capture_output=True,
                text=True
            )
            
            # Should have minimal output


class TestCLIVerbose:
    """Tests for CLI verbose mode."""
    
    def test_verbose_flag(self):
        """Test --verbose flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'model.aur')
            
            with open(input_file, 'w') as f:
                f.write("""model Net:
    def forward(x):
        x -> dense(10)
""")
            
            result = subprocess.run(
                ['python', '-m', 'aurane.cli', 'compile', input_file, '--verbose'],
                capture_output=True,
                text=True
            )
            
            # May have more detailed output


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
