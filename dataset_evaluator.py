#!/usr/bin/env python3
"""NiceGUI Dataset Evaluation Tool

A web interface to evaluate, preview, edit, and manage examples in the LoRA training dataset.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nicegui import app, ui

# Constants
DATASET_FILE = Path(__file__).parent / 'nicegui_lora_dataset.json'
BACKUP_DIR = Path(__file__).parent / 'dataset_backups'

# Global state
dataset: List[Dict[str, str]] = []
current_example_index: Optional[int] = None
preview_container = None
preview_status_label = None
instruction_input = None
response_input = None
issues_panel = None
issues_label = None
update_issues_func = None


class DatasetManager:
    """Manage dataset operations including loading, saving, and backups."""
    
    @staticmethod
    def load_dataset() -> List[Dict[str, str]]:
        """Load the dataset from JSON file."""
        try:
            with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ui.notify(f'Loaded {len(data)} examples', color='positive')
            return data
        except Exception as e:
            ui.notify(f'Error loading dataset: {e}', color='negative')
            return []
    
    @staticmethod
    def save_dataset(data: List[Dict[str, str]]) -> bool:
        """Save the dataset to JSON file with backup."""
        try:
            # Create backup
            DatasetManager.create_backup()
            
            # Save to file
            with open(DATASET_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            ui.notify('Dataset saved successfully!', color='positive')
            return True
        except Exception as e:
            ui.notify(f'Error saving dataset: {e}', color='negative')
            return False
    
    @staticmethod
    def create_backup() -> None:
        """Create a timestamped backup of the dataset."""
        BACKUP_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = BACKUP_DIR / f'dataset_backup_{timestamp}.json'
        shutil.copy2(DATASET_FILE, backup_file)
        
        # Keep only last 10 backups
        backups = sorted(BACKUP_DIR.glob('dataset_backup_*.json'))
        for old_backup in backups[:-10]:
            old_backup.unlink()
    
    @staticmethod
    def validate_example(example: Dict[str, str]) -> tuple[bool, str]:
        """Validate an example has required fields."""
        if 'instruction' not in example or not example['instruction'].strip():
            return False, "Instruction cannot be empty"
        if 'response' not in example or not example['response'].strip():
            return False, "Response cannot be empty"
        return True, ""
    
    @staticmethod
    def check_dangerous_code(code: str) -> List[str]:
        """Check for potentially dangerous operations in code."""
        warnings = []
        dangerous_patterns = [
            ('import os', 'Uses os module - may access filesystem'),
            ('import sys', 'Uses sys module - may modify system state'),
            ('import subprocess', 'Uses subprocess - may execute system commands'),
            ('open(', 'Opens files - may access filesystem'),
            ('eval(', 'Uses eval - security risk'),
            ('exec(', 'Uses exec - security risk'),
            ('__import__', 'Dynamic imports - potential security risk'),
        ]
        
        for pattern, warning in dangerous_patterns:
            if pattern in code:
                warnings.append(warning)
        
        return warnings


class ExampleListState:
    """State management for the example list."""
    search_term: str = ''
    filter_mode: str = 'all'  # 'all', 'has_issues', 'local_imports', 'generic_instructions'

example_list_state = ExampleListState()


def detect_local_imports(code: str) -> list[str]:
    """Detect imports from local files (non-standard library)."""
    import ast
    
    local_imports = []
    standard_libs = {
        'asyncio', 'base64', 'io', 'os', 'sys', 'json', 'time', 'datetime', 'pathlib',
        'typing', 'dataclasses', 'functools', 'itertools', 'collections', 're', 'math',
        'random', 'string', 'uuid', 'hashlib', 'logging', 'threading', 'multiprocessing',
    }
    common_packages = {
        'nicegui', 'fastapi', 'starlette', 'httpx', 'requests', 'numpy', 'pandas',
        'matplotlib', 'cv2', 'serial', 'zmq', 'replicate', 'openai', 'langchain',
        'langchain_openai', 'tortoise', 'stripe', 'authlib', 'descope', 'simpy',
    }
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module not in standard_libs and module not in common_packages:
                        local_imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module not in standard_libs and module not in common_packages:
                        names = ', '.join(alias.name for alias in node.names)
                        local_imports.append(f"from {node.module} import {names}")
                else:
                    # Relative import
                    names = ', '.join(alias.name for alias in node.names)
                    local_imports.append(f"from . import {names}")
    except SyntaxError:
        pass
    
    return local_imports


def has_generic_instruction(instruction: str) -> bool:
    """Check if instruction is generic (just file path)."""
    if not instruction:
        return True
    if instruction.startswith('Example from NiceGUI:'):
        return True
    if len(instruction) < 30:  # Very short instructions are likely not descriptive
        return True
    return False


@ui.refreshable
def example_list_panel():
    """Left panel showing list of examples."""
    global dataset, current_example_index
    
    search_term = example_list_state.search_term
    filter_mode = example_list_state.filter_mode
    
    # Filter examples based on search and filter mode
    filtered = []
    for idx, example in enumerate(dataset):
        instruction = example.get('instruction', '')
        response = example.get('response', '')
        
        # Apply search filter
        if search_term:
            if search_term.lower() not in instruction.lower() and search_term.lower() not in response.lower():
                continue
        
        # Apply mode filter
        if filter_mode == 'local_imports':
            local_imports = detect_local_imports(response)
            if not local_imports:
                continue
        elif filter_mode == 'generic_instructions':
            if not has_generic_instruction(instruction):
                continue
        elif filter_mode == 'has_issues':
            local_imports = detect_local_imports(response)
            generic = has_generic_instruction(instruction)
            if not local_imports and not generic:
                continue
        
        filtered.append((idx, example))
    
    if not filtered:
        ui.label('No examples found').classes('text-gray-500 mx-auto mt-4')
        return
    
    for idx, example in filtered:
        instruction = example.get('instruction', 'Untitled')
        response = example.get('response', '')
        is_selected = idx == current_example_index
        
        # Detect issues
        local_imports = detect_local_imports(response)
        generic = has_generic_instruction(instruction)
        has_shebang = response.strip().startswith('#!')
        
        # Color code based on issues
        card_classes = 'w-full cursor-pointer hover:bg-blue-50 '
        if is_selected:
            card_classes += 'bg-blue-100 border-2 border-blue-500'
        elif local_imports or generic:
            card_classes += 'border-l-4 border-yellow-500'
        
        with ui.card().classes(card_classes).on('click', lambda i=idx: load_example(i)):
            # Title with issue badges
            with ui.row().classes('w-full items-center gap-1'):
                ui.label(instruction[:60] + ('...' if len(instruction) > 60 else '')).classes('text-sm font-semibold flex-grow')
                if generic:
                    ui.badge('Generic', color='orange').classes('text-xs')
                if local_imports:
                    ui.badge('Local', color='red').classes('text-xs')
                if has_shebang:
                    ui.badge('#!', color='gray').classes('text-xs')
            
            # Show issues if any
            if local_imports:
                ui.label(f"⚠ Imports: {', '.join(local_imports[:2])}").classes('text-xs text-red-600')
            
            # Show snippet of response
            response_snippet = response[:80] + '...'
            ui.label(response_snippet).classes('text-xs text-gray-600 mt-1')


def load_example(index: int) -> None:
    """Load an example into the editor."""
    global current_example_index, dataset, instruction_input, response_input
    
    if index < 0 or index >= len(dataset):
        ui.notify('Invalid example index', color='negative')
        return
    
    if instruction_input is None or response_input is None:
        ui.notify('Editor not initialized', color='negative')
        return
    
    current_example_index = index
    example = dataset[index]
    
    instruction_input.value = example.get('instruction', '')
    response_input.value = example.get('response', '')
    
    example_list_panel.refresh()
    clear_preview()  # Clear preview instead of auto-updating
    
    # Trigger issue detection
    if update_issues_func:
        update_issues_func()
    
    ui.notify(f'Loaded example {index + 1}', color='info')


def create_new_example() -> None:
    """Create a new example."""
    global current_example_index, instruction_input, response_input
    
    current_example_index = None
    instruction_input.value = ''
    response_input.value = ''
    clear_preview()
    example_list_panel.refresh()
    ui.notify('New example created', color='info')


def evaluate_example_quality(example: Dict[str, str]) -> tuple[str, List[str]]:
    """Evaluate the quality of an example for training."""
    code = example.get('response', '')
    instruction = example.get('instruction', '')
    
    issues = []
    score = 100
    
    # Check instruction quality
    if len(instruction) < 20:
        issues.append('Instruction is too short (< 20 chars)')
        score -= 10
    
    if 'Example from NiceGUI:' in instruction and '/' in instruction:
        issues.append('Instruction is just a file path - add description')
        score -= 15
    
    # Check code quality
    if len(code) < 30:
        issues.append('Code is very short')
        score -= 15
    
    # Check for external dependencies
    analysis = analyze_code(code)
    
    if 'import' in code:
        # Check for problematic imports
        if any(imp in code for imp in ['from pathlib', 'import os', 'import sys', 'from fastapi']):
            if 'Path(__file__)' in code or 'os.path' in code:
                issues.append('Uses file paths - may not be self-contained')
                score -= 20
    
    # Check if it actually creates UI
    if not analysis['has_ui_calls'] and not analysis['needs_auto_call']:
        issues.append('No UI elements detected')
        score -= 25
    
    # Check for ui.run() which shouldn't be in examples
    if 'ui.run(' in code:
        issues.append('Contains ui.run() - not needed in examples')
        score -= 5
    
    # Good signs
    if 'ui.' in code and len(code) > 100:
        score += 5
    
    if score >= 90:
        quality = '🟢 Excellent'
    elif score >= 70:
        quality = '🟡 Good'
    elif score >= 50:
        quality = '🟠 Fair'
    else:
        quality = '🔴 Needs Improvement'
    
    return quality, issues


def save_current_example() -> None:
    """Save the current example."""
    global current_example_index, dataset, instruction_input, response_input
    
    example = {
        'instruction': instruction_input.value.strip(),
        'response': response_input.value.strip()
    }
    
    # Validate
    is_valid, error_msg = DatasetManager.validate_example(example)
    if not is_valid:
        ui.notify(error_msg, color='negative')
        return
    
    # Evaluate quality
    quality, issues = evaluate_example_quality(example)
    
    # Check for dangerous code
    warnings = DatasetManager.check_dangerous_code(example['response'])
    if warnings:
        warning_text = '\n'.join(f'⚠ {w}' for w in warnings)
        ui.notify(f'Code warnings:\n{warning_text}', color='warning')
    
    # Show quality feedback
    if issues:
        issue_text = '\n'.join(f'• {i}' for i in issues)
        ui.notify(f'Quality: {quality}\n{issue_text}', color='info' if 'Good' in quality or 'Excellent' in quality else 'warning', 
                 multi_line=True, timeout=5000)
    
    # Save or update
    if current_example_index is None:
        # Add new example
        dataset.append(example)
        current_example_index = len(dataset) - 1
        ui.notify('Example added!', color='positive')
    else:
        # Update existing
        dataset[current_example_index] = example
        ui.notify('Example updated!', color='positive')
    
    # Save to file
    DatasetManager.save_dataset(dataset)
    example_list_panel.refresh()


async def delete_current_example() -> None:
    """Delete the current example with confirmation."""
    global current_example_index, dataset
    
    if current_example_index is None:
        ui.notify('No example selected', color='warning')
        return
    
    # Confirmation dialog
    with ui.dialog() as dialog, ui.card():
        ui.label('Are you sure you want to delete this example?').classes('text-lg')
        with ui.row():
            ui.button('Cancel', on_click=dialog.close).props('outline')
            ui.button('Delete', on_click=lambda: confirm_delete(dialog), color='red')
    
    dialog.open()


def confirm_delete(dialog) -> None:
    """Confirm and execute deletion."""
    global current_example_index, dataset, instruction_input, response_input
    
    if current_example_index is not None:
        del dataset[current_example_index]
        DatasetManager.save_dataset(dataset)
        ui.notify('Example deleted', color='positive')
        
        # Clear editor
        current_example_index = None
        instruction_input.value = ''
        response_input.value = ''
        clear_preview()
        example_list_panel.refresh()
    
    dialog.close()


def analyze_code(code: str) -> dict:
    """Analyze code to detect patterns and potential issues."""
    import ast
    import re
    
    analysis = {
        'has_functions': False,
        'function_names': [],
        'has_classes': False,
        'class_names': [],
        'missing_imports': [],
        'has_ui_calls': False,
        'needs_auto_call': False,
    }
    
    try:
        tree = ast.parse(code)
        
        # Detect function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis['has_functions'] = True
                analysis['function_names'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                analysis['has_classes'] = True
                analysis['class_names'].append(node.name)
            elif isinstance(node, ast.Call):
                # Check if ui. is being called
                if hasattr(node.func, 'value') and hasattr(node.func.value, 'id'):
                    if node.func.value.id == 'ui':
                        analysis['has_ui_calls'] = True
        
        # Check if functions are called
        func_calls = re.findall(r'\b(\w+)\s*\(', code)
        called_funcs = set(func_calls)
        
        # If we have functions but no direct UI calls, might need auto-call
        if analysis['function_names'] and not analysis['has_ui_calls']:
            uncalled = [f for f in analysis['function_names'] if f not in called_funcs]
            if uncalled:
                analysis['needs_auto_call'] = True
                
    except SyntaxError:
        pass  # Will be caught during execution
    
    return analysis


def update_preview() -> None:
    """Update the live preview by executing the code."""
    global preview_container, preview_status_label
    
    code = response_input.value.strip()
    if not code:
        clear_preview()
        preview_status_label.set_text('No code to preview')
        preview_status_label.classes('text-gray-500', remove='text-green-600 text-yellow-600 text-red-600 text-blue-600')
        return
    
    # Clear previous preview
    preview_container.clear()
    
    # Analyze the code
    analysis = analyze_code(code)
    
    # Check if this uses @ui.page decorator
    has_page_decorator = '@ui.page' in code
    
    with preview_container:
        try:
            # Create a namespace for execution
            exec_namespace = {
                'ui': ui,
                'app': app,
                'Path': Path,  # Common import
            }
            
            # Remove shebang if present
            if code.startswith('#!'):
                lines = code.split('\n')
                code = '\n'.join(lines[1:])
            
            # Remove ui.run() calls as we're already running
            code_lines = []
            for line in code.split('\n'):
                if 'ui.run(' not in line and 'ui_run' not in line:
                    code_lines.append(line)
            code = '\n'.join(code_lines)
            
            # If code uses @ui.page decorator, we need to extract and call the function
            if has_page_decorator:
                # Remove @ui.page decorator and call the function directly
                modified_code = code.replace('@ui.page', '# @ui.page')
                exec(modified_code, exec_namespace)
                
                # Find and call page functions
                page_functions = [name for name in exec_namespace if callable(exec_namespace.get(name)) and name not in ['ui', 'app', 'Path']]
                
                if page_functions:
                    info_card = ui.card().classes('bg-blue-50 border-2 border-blue-300 mb-4')
                    with info_card:
                        ui.label('ℹ Page decorator detected - rendering without route').classes('text-blue-800 font-bold')
                        ui.label(f"Calling: {', '.join(page_functions)}").classes('text-blue-700 text-sm')
                    
                    for func_name in page_functions:
                        try:
                            result = exec_namespace[func_name]()
                            # If it's an async function, we can't preview it properly
                            if hasattr(result, '__await__'):
                                with ui.card().classes('bg-yellow-100 border-2 border-yellow-500'):
                                    ui.label('⚠ Async page function').classes('text-yellow-800 font-bold')
                                    ui.label('This page uses async/await which cannot be fully previewed.').classes('text-yellow-700 text-sm')
                                    ui.label('The code is valid but preview is limited.').classes('text-yellow-600 text-sm')
                        except Exception as e:
                            with ui.card().classes('bg-yellow-100 border-2 border-yellow-500'):
                                ui.label(f'Warning calling {func_name}():').classes('text-yellow-800 font-bold')
                                ui.label(str(e)).classes('text-yellow-700 font-mono text-sm')
            else:
                # Execute the code normally
                exec(code, exec_namespace)
                
                # Auto-call functions if needed
                if analysis['needs_auto_call']:
                    info_card = ui.card().classes('bg-blue-50 border-2 border-blue-300 mb-4')
                    with info_card:
                        ui.label('ℹ Auto-calling functions').classes('text-blue-800 font-bold')
                        ui.label(f"Calling: {', '.join(analysis['function_names'])}").classes('text-blue-700 text-sm')
                    
                    for func_name in analysis['function_names']:
                        if func_name in exec_namespace:
                            try:
                                exec_namespace[func_name]()
                            except Exception as e:
                                with ui.card().classes('bg-yellow-100 border-2 border-yellow-500'):
                                    ui.label(f'Warning calling {func_name}():').classes('text-yellow-800 font-bold')
                                    ui.label(str(e)).classes('text-yellow-700 font-mono text-sm')
            
            # Update status
            if analysis['has_ui_calls'] or analysis['needs_auto_call'] or has_page_decorator:
                preview_status_label.set_text('✓ Preview rendered successfully')
                preview_status_label.classes('text-green-600', remove='text-gray-500 text-yellow-600 text-red-600 text-blue-600')
            else:
                preview_status_label.set_text('ℹ Code executed (no UI elements detected)')
                preview_status_label.classes('text-blue-600', remove='text-gray-500 text-yellow-600 text-red-600 text-green-600')
            
        except ImportError as e:
            with ui.card().classes('bg-yellow-100 border-2 border-yellow-500'):
                ui.label('⚠ Import Error - External Dependency').classes('text-yellow-800 font-bold')
                ui.label(str(e)).classes('text-yellow-700 font-mono text-sm')
                ui.label('This example requires external modules/files not available in preview.').classes('text-yellow-600 text-sm')
                ui.label('Consider making examples self-contained for better training.').classes('text-yellow-600 text-sm font-semibold mt-2')
            
            preview_status_label.set_text('⚠ Has external dependencies')
            preview_status_label.classes('text-yellow-600', remove='text-gray-500 text-green-600 text-red-600 text-blue-600')
            
        except Exception as e:
            with ui.card().classes('bg-red-100 border-2 border-red-500'):
                ui.label('❌ Error in code:').classes('text-red-800 font-bold')
                ui.label(str(e)).classes('text-red-700 font-mono text-sm')
                ui.label('Check the syntax and try again.').classes('text-red-600 text-sm')
            
            preview_status_label.set_text('❌ Error in code')
            preview_status_label.classes('text-red-600', remove='text-gray-500 text-yellow-600 text-green-600 text-blue-600')


def clear_preview() -> None:
    """Clear the preview container."""
    global preview_container, preview_status_label
    if preview_container:
        preview_container.clear()
        with preview_container:
            ui.label('Preview will appear here').classes('text-gray-400 mx-auto mt-8')
    try:
        preview_status_label.set_text('No code to preview')
        preview_status_label.classes('text-gray-500', remove='text-green-600 text-yellow-600 text-red-600 text-blue-600')
    except (NameError, AttributeError):
        pass  # Status label not yet initialized


async def generate_with_lora() -> None:
    """Generate code using the LoRA model (placeholder for future implementation)."""
    with ui.dialog() as dialog, ui.card():
        ui.label('LoRA Generation').classes('text-xl font-bold')
        ui.label('This feature will integrate with your trained LoRA model.').classes('text-sm text-gray-600 mb-4')
        
        prompt_input = ui.textarea('Instruction:', placeholder='e.g., Create a button that shows a notification')
        prompt_input.classes('w-full')
        
        with ui.row():
            ui.button('Cancel', on_click=dialog.close).props('outline')
            ui.button('Generate', on_click=lambda: ui.notify('LoRA integration coming soon!', color='info'))
    
    dialog.open()


# Main UI
@ui.page('/')
def main_page():
    """Main page with the dataset evaluation interface."""
    global dataset, preview_container, preview_status_label, instruction_input, response_input
    global issues_panel, issues_label, update_issues_func
    
    # Load dataset
    dataset = DatasetManager.load_dataset()
    
    ui.colors(primary='#6E93D6', secondary='#53B689', accent='#111B1E')
    
    # Header
    with ui.header().classes('items-center'):
        ui.label('NiceGUI Dataset Evaluator').classes('text-2xl font-bold')
        ui.space()
        ui.label(f'{len(dataset)} examples').classes('text-lg')
    
    # Main content area with explicit height
    with ui.row().classes('w-full gap-2 p-2').style('height: calc(100vh - 100px)'):
        # Left panel - Example browser
        with ui.card().classes('w-80').style('height: 100%'):
            ui.label('Examples').classes('text-xl font-bold mb-2')
            
            def update_search(e):
                example_list_state.search_term = e.value
                example_list_panel.refresh()
            
            search_input = ui.input('Search...', placeholder='Search examples').classes('w-full')
            search_input.on('input', update_search)
            
            # Filter buttons
            def set_filter(mode: str):
                example_list_state.filter_mode = mode
                example_list_panel.refresh()
            
            with ui.row().classes('w-full gap-1 mt-2'):
                ui.button('All', on_click=lambda: set_filter('all'), color='primary' if example_list_state.filter_mode == 'all' else 'gray').props('dense flat').classes('text-xs')
                ui.button('Issues', on_click=lambda: set_filter('has_issues'), color='warning').props('dense flat').classes('text-xs')
                ui.button('Local', on_click=lambda: set_filter('local_imports'), color='red').props('dense flat').classes('text-xs')
                ui.button('Generic', on_click=lambda: set_filter('generic_instructions'), color='orange').props('dense flat').classes('text-xs')
            
            with ui.scroll_area().classes('w-full').style('height: calc(100% - 150px)'):
                example_list_panel()
        
        # Center panel - Code editor
        with ui.card().classes('flex-grow').style('height: 100%'):
            with ui.row().classes('w-full items-center mb-2'):
                ui.label('Editor').classes('text-xl font-bold')
                ui.space()
                issues_label = ui.label('').classes('text-sm')
            
            with ui.scroll_area().classes('w-full').style('height: calc(100% - 50px)'):
                # Issue detection panel
                issues_panel = ui.column().classes('w-full')
                
                ui.label('Instruction:').classes('font-semibold mb-1')
                instruction_input = ui.textarea(value='', placeholder='What does this example demonstrates?').classes('w-full font-mono')
                instruction_input.props('rows=3 outlined')
                
                ui.label('Response Code:').classes('font-semibold mb-1 mt-4')
                response_input = ui.textarea(value='', placeholder='NiceGUI code...').classes('w-full font-mono')
                response_input.props('rows=20 outlined')
                
                # Update issues when code changes
                def update_issues():
                    issues_panel.clear()
                    with issues_panel:
                        code = response_input.value
                        instruction = instruction_input.value
                        
                        if not code:
                            return
                        
                        local_imports = detect_local_imports(code)
                        generic = has_generic_instruction(instruction)
                        has_shebang = code.strip().startswith('#!')
                        
                        if local_imports or generic or has_shebang:
                            with ui.card().classes('bg-yellow-50 border-l-4 border-yellow-500 mb-2'):
                                ui.label('⚠ Issues Detected:').classes('font-bold text-yellow-800')
                                if generic:
                                    ui.label('• Generic instruction - add descriptive use case').classes('text-sm text-yellow-700')
                                if has_shebang:
                                    ui.label('• Has shebang line - remove #!/usr/bin/env python3').classes('text-sm text-yellow-700')
                                if local_imports:
                                    ui.label(f'• Local imports detected:').classes('text-sm text-yellow-700 font-semibold')
                                    for imp in local_imports:
                                        ui.label(f'  - {imp}').classes('text-sm text-yellow-700 font-mono ml-4')
                                    ui.label('  Consider combining with imported files').classes('text-sm text-yellow-700 ml-4')
                        
                        # Update header label
                        issue_count = len(local_imports) + (1 if generic else 0) + (1 if has_shebang else 0)
                        if issue_count > 0:
                            issues_label.set_text(f'⚠ {issue_count} issue{"s" if issue_count > 1 else ""}')
                            issues_label.classes('text-yellow-600 font-semibold', remove='text-green-600')
                        else:
                            issues_label.set_text('✓ No issues')
                            issues_label.classes('text-green-600', remove='text-yellow-600')
                
                response_input.on('blur', update_issues)
                instruction_input.on('blur', update_issues)
                
                # Make update_issues available globally
                global update_issues_func
                update_issues_func = update_issues
                
                # Action buttons
                with ui.row().classes('w-full justify-end gap-2 mt-4'):
                    ui.button('New', on_click=create_new_example, icon='add').props('outline')
                    ui.button('Delete', on_click=delete_current_example, icon='delete', color='red').props('outline')
                    ui.button('Generate with LoRA', on_click=generate_with_lora, icon='auto_awesome').props('outline')
                    ui.button('Preview', on_click=update_preview, icon='visibility', color='primary')
                    ui.button('Save', on_click=save_current_example, icon='save', color='positive')
        
        # Right panel - Live preview
        with ui.card().classes('w-[600px]').style('height: 100%'):
            with ui.row().classes('w-full items-center mb-2'):
                ui.label('Preview').classes('text-xl font-bold')
                ui.space()
                ui.button('Clear', on_click=clear_preview, icon='refresh').props('flat dense')
            
            preview_status_label = ui.label('No code to preview').classes('text-sm text-gray-500 mb-2')
            
            with ui.scroll_area().classes('w-full').style('height: calc(100% - 90px)'):
                preview_container = ui.column().classes('w-full')
                clear_preview()


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(title='NiceGUI Dataset Evaluator', port=8081, reload=False)

