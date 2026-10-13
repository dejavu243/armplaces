// Verbatim group pass from supplied Unit11.pas, lines 1530–1570.
  // Дальнейшее распределение мест
  maxW := 0; // Будем искать максимальное число побед среди тех, кто не вошел в 6-ку призеров
  for i1i := 0 to Index do if(Form1.RichEdit5.Lines[i1i] = '') and (WinCount[i1i] > maxW) then maxW := WinCount[i1i];

  Num11 := 0;  // Подсчет мест ниже 6-го
  for l := 1 to maxW + 1 do begin // С максимального для призеров 7+ кол-ва побед
    for i1i := 0 to 31 do begin // Зануляем массивы
      An[i1i] := 0;
      Bn[i1i] := 0;
    end;

    p := 6 + Num11 + 1; // Место за пределами расматриваемой группы
    k := 1;
    for i1i := 0 to Index-1 do begin
      if (WinCount[i1i] = (maxW+1-l)) and (Form1.RichEdit5.Lines[i1i] = '') then begin // Выделяем группы с одинаковым кол-вом побед
        if Form1.RichEdit5.Lines[LLos[i1i]] = '' then begin
          An[k] := p;
          p := p + 1;
        end
        else An[k] := StrToInt(Form1.RichEdit5.Lines[LLos[i1i]]);
          Bn[k] := i1i;
          k := k + 1;
        end;
    //  Form1.RichEdit2.Lines[i1i] :=  IntToStr(WinCount[i1i]);  // Кол-во побед
    //  Form1.RichEdit3.Lines[i1i] :=  Form1.RichEdit5.Lines[LLos[i1i]]; // Место последнего победившего
      end;

    WC := k-1; // Кол-во людей с одинаковым числом побед (без присужденного места)
    for i1i :=1 to WC do    // Сортировка группы (с одинаковым кол-вом побед) по местам последних выигравших их
    for j :=1 to WC-i1i do
    if An[j]>An[j+1] then begin
      p:=An[j];
      An[j]:=An[j+1]; // Массив мест победивших
      An[j+1]:=p;
      pb:=Bn[j];
      Bn[j]:=Bn[j+1]; // Массив номеров участников
      Bn[j+1]:=pb;
    end;
    for i1i :=1 to WC do Form1.RichEdit5.Lines[Bn[i1i]] := IntToStr(6 + Num11 + i1i); // Назначение мест
    Num11 := Num11 + WC;  // Сдвиг мест
  end;
