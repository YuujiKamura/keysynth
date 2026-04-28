use steel::steel_vm::engine::Engine as SteelEngine;

pub struct Engine {
    vm: SteelEngine,
}

impl Engine {
    pub fn new() -> Self {
        let mut vm = SteelEngine::new();
        let _ = vm.register_prelude();
        Self { vm }
    }

    pub fn eval(&mut self, src: &str) -> Result<String, String> {
        // Steel's Engine::run (v0.8.2) requires a &'static str for internal source tracking.
        // For Phase 1, we use Box::leak to satisfy this. This will be refactored in later phases
        // once the lifetime requirements are better understood or the engine is updated.
        let leaked_src: &'static str = Box::leak(src.to_string().into_boxed_str());

        match self.vm.run(leaked_src) {
            Ok(vals) => {
                if let Some(last) = vals.last() {
                    Ok(format!("{}", last))
                } else {
                    Ok("".to_string())
                }
            }
            Err(e) => Err(format!("{}", e)),
        }
    }
}
