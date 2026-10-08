"""Remove the unused whole-token quantization experiment from pinned runtime sources."""

def remove_token_scale(text):
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:
            raise ValueError('unexpected token-scale source: '+old[:80])
        text=text.replace(old,new,1)
    replace('pub mod int8_token_kernel;\n','')
    start=text.index('    let token_mode = cfg!(feature = "experimental-token-scale")')
    end=text.index('    let with_lora =',start)
    text=text[:start]+text[end:]
    replace('let with_lora = r.op == "lora_integer" || (token_mode && r.op == "lora_integer_token");',
            'let with_lora = r.op == "lora_integer";')
    replace('if !token_mode && n<=132','if n<=132')
    start=text.index('    let base = if token_mode {')
    end=text.index('    } else {\n        let owned;',start)
    text=text[:start]+'    let base = {\n        let owned;'+text[end+len('    } else {\n        let owned;'):]
    replace('r.op == "linear_integer_bf16" || (token_mode && r.op == "linear_integer_token_bf16")',
            'r.op == "linear_integer_bf16"')
    start=text.index('        request.op =\n            if cfg!(feature = "experimental-token-scale")')
    end=text.index('            .into();',start)+len('            .into();')
    text=text[:start]+'        request.op = "lora_integer".into();'+text[end:]
    start=text.index('    let q = if cfg!(feature = "experimental-token-scale")')
    end=text.index('    #[cfg(feature="experimental-lora-input-sharing")]',start)
    text=text[:start]+'''    let q = Some(profile::measure("activation_quantize", || {
        int8_kernel::quantize_rows(x, n, cols)
    })?);
'''+text[end:]
    replace('if r.op == "mlp_gate_up_integer"\n        || (cfg!(feature = "experimental-token-scale") && r.op == "mlp_gate_up_integer_token")',
            'if r.op == "mlp_gate_up_integer"')
    replace('''    ) || (cfg!(feature = "experimental-token-scale")
        && matches!(
            r.op.as_str(),
            "int8_matmul_token" | "linear_integer_token_bf16" | "lora_integer_token"
        ))''','    )')
    if 'experimental-token-scale' in text or 'int8_token_kernel' in text:
        raise ValueError('token-scale reference remains')
    return text
