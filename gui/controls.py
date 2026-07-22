def toggle_theme(self):
    current = ctk.get_appearance_mode()
    new_mode = "Dark" if current == "Light" else "Light"
    
    # 1. Update CustomTkinter appearance mode
    ctk.set_appearance_mode(new_mode)
    
    # 2. Update your internal theme reference & re-apply to imported modules
    self.current_theme = new_mode.lower()
    if hasattr(self, 'apply_theme'):
        self.apply_theme(self.current_theme)